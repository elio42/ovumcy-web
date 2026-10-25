package services

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ovumcy/ovumcy-web/internal/models"
)

// Prototype fork tests: tensor v1 shape, sidecar client contract, and the
// legacy fallback. Predictor is off by default, so every legacy test in the
// package runs with a nil client and never touches this path.

func predictorTestNow() (time.Time, *time.Location) {
	location := time.UTC
	return time.Date(2026, 3, 20, 12, 0, 0, 0, time.UTC), location
}

func predictorTestPeriodDay(date time.Time) models.DailyLog {
	return models.DailyLog{Date: date, IsPeriod: true, Flow: models.FlowMedium}
}

// predictorTestHistory builds three 28-day cycles ending 10 days before
// today: CompletedCycleCount 3, median 28, gates open.
func predictorTestHistory(today time.Time) []models.DailyLog {
	var logs []models.DailyLog
	for _, back := range []int{94, 93, 66, 65, 38, 37, 10, 9} {
		logs = append(logs, predictorTestPeriodDay(AddCalendarDays(today, -back, time.UTC)))
	}
	return logs
}

func predictorTestUser() *models.User {
	return &models.User{ID: 7}
}

func predictorTestReply(nextOnset, ovulation, fertileStart, fertileEnd string) string {
	reply := map[string]string{
		"next_onset": nextOnset, "ovulation_date": ovulation,
		"fertile_start": fertileStart, "fertile_end": fertileEnd,
		"model_version": "ttm-r2.1-test",
	}
	raw, _ := json.Marshal(reply)
	return string(raw)
}

func TestPredictorTensorSchemaV1(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	logs := predictorTestHistory(today)
	bbt := 36.6
	logs = append(logs, models.DailyLog{
		Date: today, BBT: &bbt, CervicalMucus: models.CervicalMucusEggWhite,
		SymptomIDs: []uint{3, 3, 5}, Flow: models.FlowHeavy,
	})

	tensor := BuildPredictorTensor(logs, predictorTestUser(), today, location)
	if tensor.Version != PredictorTensorVersion {
		t.Fatalf("tensor version = %q, want %q", tensor.Version, PredictorTensorVersion)
	}
	if len(tensor.Days) != PredictorTensorWindowDays {
		t.Fatalf("tensor days = %d, want %d", len(tensor.Days), PredictorTensorWindowDays)
	}
	for index := 1; index < len(tensor.Days); index++ {
		previous, _ := time.Parse("2006-01-02", tensor.Days[index-1].Date)
		current, _ := time.Parse("2006-01-02", tensor.Days[index].Date)
		if CalendarDaysBetween(previous, current) != 1 {
			t.Fatalf("tensor days not calendar-consecutive at %d: %s -> %s", index, tensor.Days[index-1].Date, tensor.Days[index].Date)
		}
	}
	last := tensor.Days[len(tensor.Days)-1]
	if !last.Present || last.MucusScore != 4 || last.Flow != 4 || last.SymptomCount != 2 {
		t.Fatalf("today row not encoded: %+v", last)
	}
	if last.BBT == nil || *last.BBT != bbt || last.BBTMissing {
		t.Fatalf("today BBT not carried: %+v", last)
	}
	if last.CycleDay != 11 {
		t.Fatalf("today cycle_day = %d, want 11", last.CycleDay)
	}
	if HashPredictorTensor(tensor) == "" || HashPredictorTensor(tensor) != HashPredictorTensor(tensor) {
		t.Fatal("tensor hash unstable or empty")
	}
}

func TestPredictorTensorGapAndDisturbedFlags(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	// One boundary at the window start, then silence: the tail must flag gaps.
	start := AddCalendarDays(today, -(PredictorTensorWindowDays - 1), location)
	logs := []models.DailyLog{
		predictorTestPeriodDay(start),
		predictorTestPeriodDay(AddCalendarDays(start, 1, location)),
	}
	bbt := 36.7
	logs = append(logs, models.DailyLog{
		Date: today, BBT: &bbt, CycleFactorKeys: []string{models.CycleFactorIllness},
	})
	tensor := BuildPredictorTensor(logs, predictorTestUser(), today, location)
	sawGap := false
	// Rows 0-1 are the boundary pair; the silent stretch follows them.
	for _, row := range tensor.Days[2 : len(tensor.Days)-1] {
		if row.Present {
			t.Fatalf("unexpected present row: %+v", row)
		}
		if row.GapBefore {
			sawGap = true
		}
		if !row.BBTMissing || row.MissingMask&PredictorMaskBBT == 0 {
			t.Fatalf("absent row missing BBT flags: %+v", row)
		}
	}
	if !sawGap {
		t.Fatal("no gap_before flag on a >90d silent stretch")
	}
	last := tensor.Days[len(tensor.Days)-1]
	if !last.Disturbed || !last.BBTMissing || last.BBT != nil {
		t.Fatalf("illness day must drop its BBT reading: %+v", last)
	}
}

func TestPredictorClientAppliesAndCaches(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	logs := predictorTestHistory(today)
	user := predictorTestUser()
	stats := BuildCycleStatsFromLogs(user, logs, now, location)
	legacyNext := stats.NextPeriodStart

	lastStart := AddCalendarDays(today, -10, location)
	nextOnset := CalendarDayKey(AddCalendarDays(lastStart, 30, location))
	ovulation := CalendarDayKey(AddCalendarDays(lastStart, 16, location))
	hits := 0
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		hits++
		if request.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("missing bearer token: %q", request.Header.Get("Authorization"))
		}
		var envelope struct {
			Tensor struct {
				Version string `json:"version"`
				Days    []struct {
					Date string `json:"date"`
				} `json:"days"`
			} `json:"tensor"`
			HistoryN int `json:"history_n"`
		}
		if err := json.NewDecoder(request.Body).Decode(&envelope); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if envelope.Tensor.Version != "v1" || envelope.HistoryN != 3 || len(envelope.Tensor.Days) != PredictorTensorWindowDays {
			t.Errorf("bad envelope: version=%q history_n=%d days=%d", envelope.Tensor.Version, envelope.HistoryN, len(envelope.Tensor.Days))
		}
		_, _ = writer.Write([]byte(predictorTestReply(nextOnset, ovulation, ovulation, ovulation)))
	}))
	defer server.Close()

	client := NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: server.URL, Token: "test-token", TimeoutMs: 800})
	if client == nil {
		t.Fatal("enabled client is nil")
	}
	service := NewStatsService(nil, nil)
	service.ConfigurePredictor(client)
	overlaid := BuildCycleStatsFromLogs(user, logs, now, location)
	service.ApplyPredictorOverlay(context.Background(), user, logs, &overlaid, now, location)
	if !overlaid.PredictorApplied {
		t.Fatal("neural overlay not applied")
	}
	if got := CalendarDayKey(overlaid.NextPeriodStart); got != nextOnset {
		t.Fatalf("neural next = %s, want %s (legacy %s)", got, nextOnset, CalendarDayKey(legacyNext))
	}
	if got := CalendarDayKey(overlaid.OvulationDate); got != ovulation {
		t.Fatalf("neural ovulation = %s, want %s", got, ovulation)
	}
	// Dashboard branch answers the neural pair.
	prediction := DashboardUpcomingPredictions(overlaid, user, today, DashboardProjectionCycleLength(user, overlaid))
	if got := CalendarDayKey(prediction.NextPeriodStart); got != nextOnset {
		t.Fatalf("dashboard neural next = %s, want %s", got, nextOnset)
	}
	if got := CalendarDayKey(prediction.OvulationDate); got != ovulation {
		t.Fatalf("dashboard neural ovulation = %s, want %s", got, ovulation)
	}
	// Second render serves the cache without touching the network.
	server.Close()
	cached := BuildCycleStatsFromLogs(user, logs, now, location)
	service.ApplyPredictorOverlay(context.Background(), user, logs, &cached, now, location)
	if !cached.PredictorApplied || CalendarDayKey(cached.NextPeriodStart) != nextOnset {
		t.Fatal("cache did not serve the neural dates")
	}
	if hits != 1 {
		t.Fatalf("sidecar hits = %d, want 1 (read-through cache)", hits)
	}
}

func TestPredictorFallbackKeepsLegacy(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	logs := predictorTestHistory(today)
	user := predictorTestUser()
	legacy := BuildCycleStatsFromLogs(user, logs, now, location)

	cases := map[string]func() *PredictorClient{
		"down": func() *PredictorClient {
			return NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: "http://127.0.0.1:1", TimeoutMs: 200})
		},
		"timeout": func() *PredictorClient {
			slow := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				time.Sleep(300 * time.Millisecond)
				_, _ = writer.Write([]byte(predictorTestReply("2026-04-09", "2026-03-26", "2026-03-26", "2026-03-26")))
			}))
			t.Cleanup(slow.Close)
			return NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: slow.URL, TimeoutMs: 50})
		},
		"invalid": func() *PredictorClient {
			bad := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				_, _ = writer.Write([]byte(`{"next_onset": "not-a-date"}`))
			}))
			t.Cleanup(bad.Close)
			return NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: bad.URL, TimeoutMs: 800})
		},
		"no_prediction": func() *PredictorClient {
			abstain := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				_, _ = writer.Write([]byte(`{"no_prediction": true, "reason": "thin_history"}`))
			}))
			t.Cleanup(abstain.Close)
			return NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: abstain.URL, TimeoutMs: 800})
		},
	}
	for name, build := range cases {
		t.Run(name, func(t *testing.T) {
			service := NewStatsService(nil, nil)
			service.ConfigurePredictor(build())
			got := BuildCycleStatsFromLogs(user, logs, now, location)
			service.ApplyPredictorOverlay(context.Background(), user, logs, &got, now, location)
			if got.PredictorApplied {
				t.Fatal("fallback must not apply the carrier")
			}
			if !got.NextPeriodStart.Equal(legacy.NextPeriodStart) || !got.OvulationDate.Equal(legacy.OvulationDate) {
				t.Fatalf("fallback diverged from legacy: %+v vs %+v", got, legacy)
			}
		})
	}
}

func TestPredictorRetriesOnceOn5xx(t *testing.T) {
	hits := 0
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		hits++
		if hits == 1 {
			writer.WriteHeader(http.StatusBadGateway)
			return
		}
		_, _ = writer.Write([]byte(predictorTestReply("2026-04-09", "2026-03-26", "2026-03-21", "2026-03-26")))
	}))
	defer server.Close()
	client := NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: server.URL, TimeoutMs: 800})
	tensor := BuildPredictorTensor(nil, predictorTestUser(), DateAtLocation(time.Now(), time.UTC), time.UTC)
	if _, err := client.Predict(context.Background(), 7, tensor, 0, nil); err != nil {
		t.Fatalf("retry should have recovered: %v", err)
	}
	if hits != 2 {
		t.Fatalf("hits = %d, want exactly 2 (one retry on 5xx)", hits)
	}
}

func TestPredictorStrictDecodeRefusesDrift(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		_, _ = writer.Write([]byte(`{"next_onset": "2026-04-09", "future_field": 1}`))
	}))
	defer server.Close()
	client := NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: server.URL, TimeoutMs: 800})
	tensor := BuildPredictorTensor(nil, predictorTestUser(), DateAtLocation(time.Now(), time.UTC), time.UTC)
	if _, err := client.Predict(context.Background(), 7, tensor, 0, nil); err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("unknown field must fail strict decode, got %v", err)
	}
}

func TestPredictorSkipsNetworkWhenSuppressed(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	logs := predictorTestHistory(today)
	user := predictorTestUser()
	user.UnpredictableCycle = true
	stats := BuildCycleStatsFromLogs(user, logs, now, location)
	if !PredictionsSuppressed(user, stats) {
		t.Fatal("fixture must be suppressed")
	}
	hits := 0
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		hits++
	}))
	defer server.Close()
	client := NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: server.URL, TimeoutMs: 800})
	client.ApplyPredictorOverlay(context.Background(), user, logs, &stats, now, location)
	if hits != 0 {
		t.Fatalf("suppressed tier made %d network calls, want 0", hits)
	}
	if stats.PredictorApplied {
		t.Fatal("suppressed tier must not carry neural dates")
	}
}

func TestPredictorClampsNarrowWindow(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	logs := predictorTestHistory(today)
	user := predictorTestUser()
	legacy := BuildCycleStatsFromLogs(user, logs, now, location)
	legacyWidth := CalendarDaysBetween(legacy.FertilityWindowStart, legacy.FertilityWindowEnd)
	if legacyWidth <= 0 {
		t.Fatalf("fixture has no legacy window width: %+v", legacy)
	}
	lastStart := AddCalendarDays(today, -10, location)
	nextOnset := CalendarDayKey(AddCalendarDays(lastStart, 30, location))
	ovulation := CalendarDayKey(AddCalendarDays(lastStart, 16, location))
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		// Overconfident one-day window: must widen to at least legacy width.
		_, _ = writer.Write([]byte(predictorTestReply(nextOnset, ovulation, ovulation, ovulation)))
	}))
	defer server.Close()
	service := NewStatsService(nil, nil)
	service.ConfigurePredictor(NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: server.URL, TimeoutMs: 800}))
	got := BuildCycleStatsFromLogs(user, logs, now, location)
	service.ApplyPredictorOverlay(context.Background(), user, logs, &got, now, location)
	if width := CalendarDaysBetween(got.FertilityWindowStart, got.FertilityWindowEnd); width < legacyWidth {
		t.Fatalf("neural window width %d below legacy %d", width, legacyWidth)
	}
	if got.FertilityWindowEnd.IsZero() || CalendarDayKey(got.FertilityWindowEnd) != ovulation {
		t.Fatalf("clamped window must still end on the neural ovulation: %+v", got)
	}
}

func TestPredictorDisabledByDefault(t *testing.T) {
	if NewPredictorClient(PredictorConfig{}) != nil {
		t.Fatal("zero config must yield a nil client")
	}
	if NewPredictorClient(PredictorConfig{Enabled: true}) != nil {
		t.Fatal("enabled config without a URL must yield a nil client")
	}
	var nilClient *PredictorClient
	nilClient.ApplyPredictorOverlay(context.Background(), predictorTestUser(), nil, &CycleStats{}, time.Now(), time.UTC)
}
