package services

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/ovumcy/ovumcy-web/internal/models"
)

// Transformer sidecar client (prototype fork). This file is the ONLY network
// touchpoint for predictions: services otherwise return domain results and
// never dial out for them (webhook delivery aside). All configuration arrives
// through PredictorConfig constructor args from bootstrap.Options — never
// globals, never os.Getenv deep in services.

// DefaultPredictorTimeoutMs bounds one sidecar attempt. A slow sidecar must
// never stall a page render; on timeout the caller keeps the legacy median.
const DefaultPredictorTimeoutMs = 800

// predictorCacheTTL bounds a cached sidecar answer; predictorCacheMaxEntries
// bounds the map itself (prototype in-memory read-through: a restart drops
// it, and the next render re-fetches).
const (
	predictorCacheTTL        = 24 * time.Hour
	predictorCacheMaxEntries = 1024
)

// PredictorConfig wires the sidecar. Enabled is the kill switch (default
// off); BaseURL names the shared-Docker-net origin (no TLS on that net by
// design, bearer token only); Token is the shared bearer secret.
// TensorVersion selects the tensor schema the client builds: "v2" routes to
// the mixed-v2 sidecar (14 columns incl. libido peak); anything else builds
// v1, so stub/v1/legacy stacks keep working with zero behavior change.
type PredictorConfig struct {
	Enabled       bool
	BaseURL       string
	Token         string
	TimeoutMs     int
	TensorVersion string
}

// PredictorTensorVersion answers the tensor schema version this client
// builds: exactly "v2" when configured, "v1" otherwise (empty included).
func (config PredictorConfig) PredictorTensorVersion() string {
	if config.TensorVersion == PredictorTensorVersionV2 {
		return PredictorTensorVersionV2
	}
	return PredictorTensorVersion
}

// PredictorTimeout answers the per-attempt timeout, falling back to the
// default when unset or non-positive.
func (config PredictorConfig) PredictorTimeout() time.Duration {
	if config.TimeoutMs <= 0 {
		return DefaultPredictorTimeoutMs * time.Millisecond
	}
	return time.Duration(config.TimeoutMs) * time.Millisecond
}

// predictorRequest is the POST /v1/predict body. Tensor and history are the
// same daily series and recent lengths the App side builds; the sidecar
// decides zero-shot vs finetuned from history_n.
type predictorRequest struct {
	UserID        uint            `json:"user_id"`
	Tensor        PredictorTensor `json:"tensor"`
	HistoryN      int             `json:"history_n"`
	RecentLengths []int           `json:"recent_lengths"`
}

// predictorResponse is the strict sidecar reply. no_prediction is the
// sidecar's own abstention (thin history it will not call): the app keeps
// the legacy median and renders existing empty states.
type predictorResponse struct {
	NoPrediction  bool   `json:"no_prediction"`
	Reason        string `json:"reason"`
	NextOnset     string `json:"next_onset"`
	OvulationDate string `json:"ovulation_date"`
	FertileStart  string `json:"fertile_start"`
	FertileEnd    string `json:"fertile_end"`
	P10           string `json:"p10"`
	P50           string `json:"p50"`
	P90           string `json:"p90"`
	ModelVersion  string `json:"model_version"`
}

// PredictorResult is a decoded, accepted sidecar answer. Dates are
// UTC-midnight date-only values, the same shape PredictCycleWindow returns;
// the overlay converts them onto the request location.
type PredictorResult struct {
	NextOnset    time.Time
	Ovulation    time.Time
	FertileStart time.Time
	FertileEnd   time.Time
	ModelVersion string
	LatencyMs    int64
}

type predictorCacheEntry struct {
	result   PredictorResult
	storedAt time.Time
}

// PredictorClient posts tensors to the sidecar with a read-through cache.
// The zero value is disabled; NewPredictorClient returns nil for a disabled
// or unconfigured config so a nil *PredictorClient reads as "legacy only".
type PredictorClient struct {
	config PredictorConfig
	http   *http.Client
	mutex  sync.Mutex
	cache  map[string]predictorCacheEntry
}

// NewPredictorClient builds the sidecar client, or nil when the fork is off.
func NewPredictorClient(config PredictorConfig) *PredictorClient {
	if !config.Enabled || config.BaseURL == "" {
		return nil
	}
	return &PredictorClient{config: config, http: &http.Client{}, cache: make(map[string]predictorCacheEntry)}
}

// errPredictorNoPrediction marks the sidecar's own abstention: not a failure,
// the caller keeps the legacy median with no banner.
var errPredictorNoPrediction = fmt.Errorf("predictor: no_prediction")

// hashPredictorUser answers the log-safe owner handle: a hash, never the id.
func hashPredictorUser(userID uint) string {
	sum := sha256.Sum256([]byte("predictor-user:" + strconv.FormatUint(uint64(userID), 10)))
	return hex.EncodeToString(sum[:])
}

// Predict posts one tensor and decodes the strict reply. Exactly one retry on
// 5xx; timeouts, connection failures and invalid bodies fall back with no
// retry. Raw BBT never reaches logs or errors — only the user hash, the input
// hash, the model version and the latency.
func (client *PredictorClient) Predict(ctx context.Context, userID uint, tensor PredictorTensor, historyN int, recentLengths []int) (PredictorResult, error) {
	if client == nil {
		return PredictorResult{}, fmt.Errorf("predictor: disabled")
	}
	inputHash := HashPredictorTensor(tensor)
	userHash := hashPredictorUser(userID)
	cacheKey := strconv.FormatUint(uint64(userID), 10) + ":" + inputHash
	if cached, ok := client.cached(cacheKey); ok {
		return cached, nil
	}
	body, err := json.Marshal(predictorRequest{UserID: userID, Tensor: tensor, HistoryN: historyN, RecentLengths: recentLengths})
	if err != nil {
		return PredictorResult{}, fmt.Errorf("predictor: encode: %w", err)
	}
	start := time.Now()
	result, err := client.post(ctx, body)
	latencyMs := time.Since(start).Milliseconds()
	if err != nil {
		log.Printf("predictor: fallback to legacy (user=%s input=%s err=%v)", userHash, inputHash, err)
		return PredictorResult{}, err
	}
	result.LatencyMs = latencyMs
	log.Printf("predictor: applied model=%s latency_ms=%d user=%s input=%s", result.ModelVersion, latencyMs, userHash, inputHash)
	client.store(cacheKey, result)
	return result, nil
}

func (client *PredictorClient) cached(key string) (PredictorResult, bool) {
	client.mutex.Lock()
	defer client.mutex.Unlock()
	entry, ok := client.cache[key]
	if !ok || time.Since(entry.storedAt) > predictorCacheTTL {
		return PredictorResult{}, false
	}
	return entry.result, true
}

func (client *PredictorClient) store(key string, result PredictorResult) {
	client.mutex.Lock()
	defer client.mutex.Unlock()
	if len(client.cache) >= predictorCacheMaxEntries {
		client.cache = make(map[string]predictorCacheEntry, predictorCacheMaxEntries)
	}
	client.cache[key] = predictorCacheEntry{result: result, storedAt: time.Now()}
}

func (client *PredictorClient) post(ctx context.Context, body []byte) (PredictorResult, error) {
	endpoint := client.config.BaseURL + "/v1/predict"
	attempt := func() (*http.Response, error) {
		attemptCtx, cancel := context.WithTimeout(ctx, client.config.PredictorTimeout())
		defer cancel()
		request, err := http.NewRequestWithContext(attemptCtx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Accept", "application/json")
		if client.config.Token != "" {
			request.Header.Set("Authorization", "Bearer "+client.config.Token)
		}
		return client.http.Do(request)
	}
	response, err := attempt()
	if err != nil {
		return PredictorResult{}, err
	}
	defer response.Body.Close()
	if response.StatusCode >= 500 && response.StatusCode <= 599 {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 1<<20))
		response.Body.Close()
		response, err = attempt()
		if err != nil {
			return PredictorResult{}, err
		}
		defer response.Body.Close()
	}
	if response.StatusCode < 200 || response.StatusCode > 299 {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 1<<20))
		return PredictorResult{}, fmt.Errorf("predictor: status %d", response.StatusCode)
	}
	return decodePredictorResponse(io.LimitReader(response.Body, 1<<20))
}

// decodePredictorResponse strictly decodes one sidecar reply: unknown fields
// are refused so a contract drift fails loudly into the legacy fallback.
func decodePredictorResponse(reader io.Reader) (PredictorResult, error) {
	var reply predictorResponse
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&reply); err != nil {
		return PredictorResult{}, fmt.Errorf("predictor: decode: %w", err)
	}
	if reply.NoPrediction {
		return PredictorResult{}, errPredictorNoPrediction
	}
	result := PredictorResult{ModelVersion: reply.ModelVersion}
	var err error
	if result.NextOnset, err = parsePredictorDay(reply.NextOnset, "next_onset"); err != nil {
		return PredictorResult{}, err
	}
	if result.NextOnset.IsZero() {
		return PredictorResult{}, fmt.Errorf("predictor: empty next_onset")
	}
	if result.Ovulation, err = parsePredictorDay(reply.OvulationDate, "ovulation_date"); err != nil {
		return PredictorResult{}, err
	}
	if result.FertileStart, err = parsePredictorDay(reply.FertileStart, "fertile_start"); err != nil {
		return PredictorResult{}, err
	}
	if result.FertileEnd, err = parsePredictorDay(reply.FertileEnd, "fertile_end"); err != nil {
		return PredictorResult{}, err
	}
	return result, nil
}

// parsePredictorDay answers a YYYY-MM-DD sidecar date as a UTC-midnight
// date-only value; empty means absent, anything else malformed is an error.
func parsePredictorDay(raw string, field string) (time.Time, error) {
	if raw == "" {
		return time.Time{}, nil
	}
	day, err := time.Parse("2006-01-02", raw)
	if err != nil {
		return time.Time{}, fmt.Errorf("predictor: invalid %s: %w", field, err)
	}
	return day, nil
}

// PredictorStatsOverlay is the optional sidecar capability
// (prototype fork): StatsService implements it, and the dashboard builder
// asserts for it so test fakes without a client keep the legacy path without
// a signature change.
type PredictorStatsOverlay interface {
	ApplyPredictorOverlay(ctx context.Context, user *models.User, logs []models.DailyLog, stats *CycleStats, now time.Time, location *time.Location)
}

// ApplyPredictorOverlay tries the sidecar for one stats derivation and
// overlays the neural dates onto stats, keeping the legacy median on any
// failure. Gates run BEFORE any network I/O: a suppressed whole-projection
// skips the call entirely, and the fertility half overlays only where the
// fertility gate is open. Narrow neural windows widen to at least the legacy
// width. The ctx carries the caller's deadline (never re-rooted: the
// context-rooting guard refuses a detached context here); the package-level
// derivation stays pure legacy.
func (client *PredictorClient) ApplyPredictorOverlay(ctx context.Context, user *models.User, logs []models.DailyLog, stats *CycleStats, now time.Time, location *time.Location) {
	if client == nil || user == nil || stats == nil {
		return
	}
	if location == nil {
		location = time.UTC
	}
	if PredictionsSuppressed(user, *stats) {
		return
	}
	today := DateAtLocation(now, location)
	tensor := BuildPredictorTensor(logs, user, today, location)
	if client.config.PredictorTensorVersion() == PredictorTensorVersionV2 {
		tensor = BuildPredictorTensorV2(logs, user, today, location)
	}
	starts := CycleBoundaries(filterLogsNotAfter(logs, today), BoundaryContextFor(user, today))
	recentLengths := predictorRecentLengths(starts)
	ctx, cancel := context.WithTimeout(ctx, client.config.PredictorTimeout())
	defer cancel()
	result, err := client.Predict(ctx, user.ID, tensor, stats.CompletedCycleCount, recentLengths)
	if err != nil {
		return
	}
	next := projectedDay(CalendarDay(result.NextOnset, location))
	if next.IsZero() {
		return
	}
	stats.NextPeriodStart = next
	stats.PredictorApplied = true
	stats.PredictorModelVersion = result.ModelVersion
	stats.PredictorNeuralNextPeriodStart = CalendarDay(result.NextOnset, location)
	stats.PredictorNeuralOvulationDate = CalendarDay(result.Ovulation, location)
	stats.PredictorNeuralFertileStart = CalendarDay(result.FertileStart, location)
	stats.PredictorNeuralFertileEnd = CalendarDay(result.FertileEnd, location)
	if result.Ovulation.IsZero() || FertilityProjectionSuppressed(user, *stats) {
		stats.PredictorNeuralOvulationDate = time.Time{}
		stats.PredictorNeuralFertileStart = time.Time{}
		stats.PredictorNeuralFertileEnd = time.Time{}
		return
	}
	legacy := PredictCycleWindow(stats.LastPeriodStart, predictedCycleLength(stats.MedianCycleLength, stats.AverageCycleLength), stats.LutealPhase)
	fertileStart, fertileEnd := predictorClampedWindow(result, legacy, stats.LastPeriodStart, location)
	stats.OvulationDate = CalendarDay(result.Ovulation, location)
	stats.OvulationExact = false
	stats.OvulationImpossible = false
	stats.FertilityWindowStart = fertileStart
	stats.FertilityWindowEnd = fertileEnd
}

// predictorRecentLengths answers the recent cycle lengths for the sidecar
// envelope, newest last, over the same recent window the median uses.
func predictorRecentLengths(starts []time.Time) []int {
	if len(starts) < 2 {
		return nil
	}
	lengths := make([]int, 0, len(starts)-1)
	for index := 1; index < len(starts); index++ {
		lengths = append(lengths, CalendarDaysBetween(starts[index-1], starts[index]))
	}
	return tailInts(lengths, cyclePredictionWindow)
}

// predictorClampedWindow answers the neural fertile window widened to at
// least the legacy width: a sidecar that returns an overconfident narrow
// window must not shrink what the owner sees below the irregular spread.
func predictorClampedWindow(result PredictorResult, legacy CycleWindowPrediction, lastPeriodStart time.Time, location *time.Location) (time.Time, time.Time) {
	ovulation := CalendarDay(result.Ovulation, location)
	anchor := CalendarDay(lastPeriodStart, location)
	start := CalendarDay(result.FertileStart, location)
	if start.IsZero() {
		start = AddCalendarDays(ovulation, -5, location)
	}
	if !anchor.IsZero() && start.Before(anchor) {
		start = anchor
	}
	end := CalendarDay(result.FertileEnd, location)
	if end.IsZero() || end.Before(ovulation) {
		end = ovulation
	}
	if legacy.Calculable {
		legacyWidth := CalendarDaysBetween(legacy.FertilityWindowStart, legacy.FertilityWindowEnd)
		if CalendarDaysBetween(start, end) < legacyWidth {
			start = AddCalendarDays(end, -legacyWidth, location)
			if !anchor.IsZero() && start.Before(anchor) {
				start = anchor
			}
		}
	}
	return start, end
}

// neuralUpcomingPrediction answers the neural next-period/ovulation pair for
// the single DashboardUpcomingPredictions branch: the carrier
// ApplyPredictorOverlay wrote beside the legacy stats. Absent carrier
// (legacy path, suppressed tiers, failed sidecar) answers false and the
// legacy math runs untouched.
func neuralUpcomingPrediction(stats CycleStats, today time.Time, cycleLength int) (DashboardUpcomingPrediction, bool) {
	absent := DashboardUpcomingPrediction{}
	if !stats.PredictorApplied || stats.PredictorNeuralNextPeriodStart.IsZero() {
		return absent, false
	}
	if stats.LastPeriodStart.IsZero() || cycleLength <= 0 {
		return absent, false
	}
	if _, _, ok := ProjectCycleStart(stats.LastPeriodStart, cycleLength, today); !ok {
		return absent, false
	}
	location := today.Location()
	if location == nil {
		location = time.UTC
	}
	next := projectedDay(CalendarDay(stats.PredictorNeuralNextPeriodStart, location))
	if next.IsZero() {
		return absent, false
	}
	out := DashboardUpcomingPrediction{NextPeriodStart: next, OvulationImpossible: true}
	if stats.PredictorNeuralOvulationDate.IsZero() {
		return out, true
	}
	ovulation := projectedDay(CalendarDay(stats.PredictorNeuralOvulationDate, location))
	if ovulation.IsZero() {
		return out, true
	}
	out.OvulationDate = ovulation
	out.OvulationExact = false
	out.OvulationImpossible = false
	return out, true
}
