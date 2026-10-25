package services

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"time"

	"github.com/ovumcy/ovumcy-web/internal/models"
)

// Daily tensor v1 for the transformer sidecar (prototype fork).
//
// The App side builds this tensor and the training repo consumes it; the two
// must stay byte-identical (see docs/predictor-contract.md). The builder is a
// pure function over already-fetched logs: no repositories, no network, no
// clock reads — the caller supplies today and the location.

// PredictorTensorVersion is the schema version every tensor carries. The
// sidecar routes on it; a v2 schema is a new version, never an edit of v1.
const PredictorTensorVersion = "v1"

// PredictorTensorVersionV2 is the v2 schema version: v1 rows plus the
// libido-peak columns (peak, peak_missing). The v2 builder is additive —
// BuildPredictorTensor stays byte-identical and untouched.
const PredictorTensorVersionV2 = "v2"

// PredictorTensorWindowDays is the daily history window the tensor covers,
// ending on the owner's today. 180 days keeps short histories whole while
// giving the transformer the 90+ daily points it needs (a cycle-level series
// is only 3-50 points and too short for attention).
const PredictorTensorWindowDays = 180

// PredictorSkipGapDays is the span past which missing days are flagged as a
// gap, not averaged. A missed period log merges two real cycles into one
// ~60-90 day gap; the tensor marks days_since_last_log > 90 so the sidecar
// treats the stretch as skips (k=2) instead of one long cycle.
const PredictorSkipGapDays = 90

// PredictorDayRow is one calendar day of the v1 tensor. Present days carry
// the owner's signals; absent days carry only flags, and the missingness mask
// says which fields are real. BBT is the stored Celsius value or nil — never
// a sentinel — with bbt_missing naming the absence.
type PredictorDayRow struct {
	Date         string   `json:"date"`
	Present      bool     `json:"present"`
	Bleed        int      `json:"bleed"`
	Flow         int      `json:"flow"`
	BBT          *float64 `json:"bbt_c"`
	BBTMissing   bool     `json:"bbt_missing"`
	MucusScore   int      `json:"mucus_score"`
	SymptomCount int      `json:"symptom_count"`
	Disturbed    bool     `json:"disturbed"`
	CycleDay     int      `json:"cycle_day"`
	GapBefore    bool     `json:"gap_before"`
	MissingMask  int      `json:"missing_mask"`
	// Peak and PeakMissing are the v2 libido-peak columns (ints like
	// Bleed/Flow, matching the Python wire keys the matrix builder reads
	// with defaults). The v1 builder leaves them zero; the v1 sidecar
	// ignores unknown request fields, so v1 serving is unaffected.
	Peak        int `json:"peak"`
	PeakMissing int `json:"peak_missing"`
}

// PredictorTensor is the v1 request payload beside the envelope the client
// adds (user_id, history_n, recent lengths).
type PredictorTensor struct {
	Version string            `json:"version"`
	Days    []PredictorDayRow `json:"days"`
}

// Missing-mask bits for PredictorDayRow.MissingMask.
const (
	PredictorMaskAbsentRow = 1 << iota
	PredictorMaskBBT
	PredictorMaskCycleDay
)

// BuildPredictorTensor lays the owner's logs onto one row per calendar day
// over the [today-window+1, today] window. Day steps walk through
// AddCalendarDays and cycle days measure through CalendarDaysBetween, so a
// DST-skipped midnight never shifts a row (issue #48 class). Cycle days read
// the same CycleBoundaries every other surface reads; a day before the first
// known boundary carries cycle_day 0 with the cycle-day mask bit set.
func BuildPredictorTensor(logs []models.DailyLog, user *models.User, today time.Time, location *time.Location) PredictorTensor {
	if location == nil {
		location = time.UTC
	}
	today = CalendarDay(today, location)
	boundaries := CycleBoundaries(filterLogsNotAfter(logs, today), BoundaryContextFor(user, today))

	byDay := make(map[string]models.DailyLog, len(logs))
	for _, entry := range sortDailyLogs(logs) {
		day := CalendarDay(entry.Date, location)
		if day.IsZero() || CalendarDaysBetween(day, today) < 0 {
			continue
		}
		// The latest same-day write wins, matching the chart series.
		byDay[CalendarDayKey(day)] = entry
	}

	start := AddCalendarDays(today, -(PredictorTensorWindowDays - 1), location)
	tensor := PredictorTensor{Version: PredictorTensorVersion, Days: make([]PredictorDayRow, 0, PredictorTensorWindowDays)}
	daysSincePresent := -1
	for day := start; !day.IsZero() && CalendarDaysBetween(day, today) >= 0; day = AddCalendarDays(day, 1, location) {
		key := CalendarDayKey(day)
		entry, present := byDay[key]
		row := PredictorDayRow{Date: key, Present: present}
		if !present {
			row.MissingMask = PredictorMaskAbsentRow | PredictorMaskBBT
			row.BBTMissing = true
			if daysSincePresent > PredictorSkipGapDays {
				row.GapBefore = true
			}
			if daysSincePresent >= 0 {
				daysSincePresent++
			}
			row.CycleDay = predictorCycleDay(boundaries, day)
			if row.CycleDay <= 0 {
				row.MissingMask |= PredictorMaskCycleDay
			}
			tensor.Days = append(tensor.Days, row)
			continue
		}
		daysSincePresent = 0
		mask := 0
		row.Bleed = predictorBleed(entry)
		row.Flow = predictorFlowScore(entry.Flow)
		if entry.BBT != nil && IsValidDayBBT(entry.BBT) && !isBBTDisturbedLog(entry) {
			value := *entry.BBT
			row.BBT = &value
		} else {
			row.BBTMissing = true
			mask |= PredictorMaskBBT
		}
		row.MucusScore = predictorMucusScore(entry.CervicalMucus)
		row.SymptomCount = len(uniqueSymptomIDs(entry.SymptomIDs))
		row.Disturbed = isBBTDisturbedLog(entry)
		row.CycleDay = predictorCycleDay(boundaries, day)
		if row.CycleDay <= 0 {
			mask |= PredictorMaskCycleDay
		}
		row.MissingMask = mask
		tensor.Days = append(tensor.Days, row)
	}
	return tensor
}

// BuildPredictorTensorV2 answers the v2 tensor: the v1 rows plus the
// tracking-aware libido-peak columns (peak, peak_missing), carrying
// Version "v2" so the sidecar routes it to the 14-column matrix.
//
// Mapping (window = the same 180d byDay map the v1 builder uses):
//   - active = any byDay entry with LibidoPeak set (the owner tracks libido).
//   - present + checked          -> peak=1, peak_missing=0 (observed peak).
//   - present + unchecked + active   -> peak=0, peak_missing=0 (known-absent).
//   - present + unchecked + inactive -> peak=0, peak_missing=1 (unknown:
//     an unchecked box from a never-tracking owner is not known-absent;
//     emitting known-absent everywhere would mimic anovulatory Fehring
//     cycles, a regime the model never saw on ovulatory windows).
//   - absent -> peak=0, peak_missing=1 (same as Python build_tensor_v2).
//
// The v1 rows are built by BuildPredictorTensor itself, so the first 12
// fields stay identical to v1 by construction; only Version and the two
// peak columns differ.
func BuildPredictorTensorV2(logs []models.DailyLog, user *models.User, today time.Time, location *time.Location) PredictorTensor {
	tensor := BuildPredictorTensor(logs, user, today, location)
	tensor.Version = PredictorTensorVersionV2
	if location == nil {
		location = time.UTC
	}
	today = CalendarDay(today, location)
	byDay := make(map[string]models.DailyLog, len(logs))
	for _, entry := range sortDailyLogs(logs) {
		day := CalendarDay(entry.Date, location)
		if day.IsZero() || CalendarDaysBetween(day, today) < 0 {
			continue
		}
		// The latest same-day write wins, matching the v1 builder.
		byDay[CalendarDayKey(day)] = entry
	}
	active := false
	for _, entry := range byDay {
		if entry.LibidoPeak {
			active = true
			break
		}
	}
	for index, row := range tensor.Days {
		if !row.Present {
			tensor.Days[index].Peak = 0
			tensor.Days[index].PeakMissing = 1
			continue
		}
		entry := byDay[row.Date]
		switch {
		case entry.LibidoPeak:
			tensor.Days[index].Peak = 1
			tensor.Days[index].PeakMissing = 0
		case active:
			tensor.Days[index].Peak = 0
			tensor.Days[index].PeakMissing = 0
		default:
			tensor.Days[index].Peak = 0
			tensor.Days[index].PeakMissing = 1
		}
	}
	return tensor
}

// predictorCycleDay answers the 1-based cycle day of day under the shared
// boundary rule, or 0 when no boundary is on or before it.
func predictorCycleDay(boundaries []time.Time, day time.Time) int {
	anchor := latestBoundaryOnOrBefore(boundaries, day)
	if anchor.IsZero() {
		return 0
	}
	return CalendarDaysBetween(anchor, day) + 1
}

// predictorBleed folds a log to the tensor's bleed bit: any non-spotting
// period signal, matching the boundary rule's reading (spotting never opens
// a cycle, so it never reads as bleed here either).
func predictorBleed(entry models.DailyLog) int {
	if !entry.IsPeriod {
		return 0
	}
	if isSpottingDay(entry) {
		return 0
	}
	return 1
}

// predictorFlowScore folds a stored flow to the v1 ordinal.
func predictorFlowScore(flow string) int {
	switch NormalizeDayFlow(flow) {
	case models.FlowSpotting:
		return 1
	case models.FlowLight:
		return 2
	case models.FlowMedium:
		return 3
	case models.FlowHeavy:
		return 4
	default:
		return 0
	}
}

// predictorMucusScore folds cervical mucus to the v1 ordinal, peak last.
func predictorMucusScore(value string) int {
	switch NormalizeDayCervicalMucus(value) {
	case models.CervicalMucusDry:
		return 1
	case models.CervicalMucusMoist:
		return 2
	case models.CervicalMucusCreamy:
		return 3
	case models.CervicalMucusEggWhite:
		return 4
	default:
		return 0
	}
}

// HashPredictorTensor answers the canonical-input hash the client caches on:
// sha256 over the versioned tensor JSON. Raw BBT never reaches logs — only
// this hash does.
func HashPredictorTensor(tensor PredictorTensor) string {
	raw, err := json.Marshal(tensor)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
