package services

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ovumcy/ovumcy-web/internal/models"
)

// Prototype fork tests: tensor v2 (libido-peak columns) and the v2 overlay
// route. The v1 builder stays byte-identical; v2 is additive.

// predictorV2RowByDate indexes v2 tensor days by date for golden lookups.
func predictorV2RowByDate(t *testing.T, tensor PredictorTensor) map[string]PredictorDayRow {
	t.Helper()
	rows := make(map[string]PredictorDayRow, len(tensor.Days))
	for _, row := range tensor.Days {
		rows[row.Date] = row
	}
	return rows
}

func TestPredictorTensorV2VersionAndShape(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	logs := predictorTestHistory(today)

	tensor := BuildPredictorTensorV2(logs, predictorTestUser(), today, location)
	if tensor.Version != PredictorTensorVersionV2 {
		t.Fatalf("tensor version = %q, want %q", tensor.Version, PredictorTensorVersionV2)
	}
	if tensor.Version == PredictorTensorVersion {
		t.Fatal("v2 tensor must not carry the v1 version")
	}
	if len(tensor.Days) != PredictorTensorWindowDays {
		t.Fatalf("tensor days = %d, want %d", len(tensor.Days), PredictorTensorWindowDays)
	}
}

func TestPredictorTensorV2InactiveTrackerIsUnknown(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	// No LibidoPeak anywhere: the owner never tracks libido, so every
	// unchecked row — present or absent — is unknown, never known-absent.
	logs := predictorTestHistory(today)

	tensor := BuildPredictorTensorV2(logs, predictorTestUser(), today, location)
	sawPresent := false
	for _, row := range tensor.Days {
		if row.Peak != 0 || row.PeakMissing != 1 {
			t.Fatalf("inactive tracker row must be (0,1): %+v", row)
		}
		if row.Present {
			sawPresent = true
		}
		if !row.Present && (row.Peak != 0 || row.PeakMissing != 1) {
			t.Fatalf("absent row must be (0,1): %+v", row)
		}
	}
	if !sawPresent {
		t.Fatal("fixture has no present rows to check")
	}
}

func TestPredictorTensorV2ActiveTrackerMapping(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	peakDay := AddCalendarDays(today, -20, location)
	logs := append(predictorTestHistory(today), models.DailyLog{
		Date: peakDay, LibidoPeak: true,
	})

	tensor := BuildPredictorTensorV2(logs, predictorTestUser(), today, location)
	rows := predictorV2RowByDate(t, tensor)

	checked, ok := rows[CalendarDayKey(peakDay)]
	if !ok {
		t.Fatal("peak day missing from tensor")
	}
	if !checked.Present || checked.Peak != 1 || checked.PeakMissing != 0 {
		t.Fatalf("checked peak day must be (1,0): %+v", checked)
	}
	// Active tracker: other present-but-unchecked rows are known-absent.
	for _, row := range tensor.Days {
		if !row.Present || row.Date == checked.Date {
			continue
		}
		if row.Peak != 0 || row.PeakMissing != 0 {
			t.Fatalf("active-tracker unchecked row must be (0,0): %+v", row)
		}
	}
	// Absent rows stay unknown even for an active tracker.
	for _, row := range tensor.Days {
		if row.Present {
			continue
		}
		if row.Peak != 0 || row.PeakMissing != 1 {
			t.Fatalf("absent row must be (0,1): %+v", row)
		}
	}
}

func TestPredictorTensorV2FirstTwelveMatchV1(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	peakDay := AddCalendarDays(today, -20, location)
	logs := append(predictorTestHistory(today),
		models.DailyLog{Date: peakDay, LibidoPeak: true},
		models.DailyLog{Date: today, CervicalMucus: models.CervicalMucusEggWhite},
	)

	user := predictorTestUser()
	v1 := BuildPredictorTensor(logs, user, today, location)
	v2 := BuildPredictorTensorV2(logs, user, today, location)
	if len(v1.Days) != len(v2.Days) {
		t.Fatalf("day count diverged: v1=%d v2=%d", len(v1.Days), len(v2.Days))
	}
	for index := range v1.Days {
		a, b := v1.Days[index], v2.Days[index]
		if a.Date != b.Date || a.Present != b.Present || a.Bleed != b.Bleed ||
			a.Flow != b.Flow || a.BBTMissing != b.BBTMissing || a.MucusScore != b.MucusScore ||
			a.SymptomCount != b.SymptomCount || a.Disturbed != b.Disturbed ||
			a.CycleDay != b.CycleDay || a.GapBefore != b.GapBefore || a.MissingMask != b.MissingMask {
			t.Fatalf("v2 row %d diverges from v1 in the first 12 fields:\nv1=%+v\nv2=%+v", index, a, b)
		}
		if (a.BBT == nil) != (b.BBT == nil) || (a.BBT != nil && *a.BBT != *b.BBT) {
			t.Fatalf("v2 row %d BBT diverges from v1: %v vs %v", index, a.BBT, b.BBT)
		}
	}
}

func TestPredictorTensorVersionDefaultsToV1(t *testing.T) {
	for _, version := range []string{"", "v1", "V2", "v3"} {
		config := PredictorConfig{Enabled: true, BaseURL: "http://predictor:8000", TensorVersion: version}
		if got := config.PredictorTensorVersion(); got != PredictorTensorVersion {
			t.Fatalf("TensorVersion %q resolved to %q, want v1", version, got)
		}
	}
	config := PredictorConfig{Enabled: true, BaseURL: "http://predictor:8000", TensorVersion: "v2"}
	if got := config.PredictorTensorVersion(); got != PredictorTensorVersionV2 {
		t.Fatalf("TensorVersion v2 resolved to %q", got)
	}
}

func TestPredictorV2OverlayApplies(t *testing.T) {
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	peakDay := AddCalendarDays(today, -20, location)
	logs := append(predictorTestHistory(today), models.DailyLog{
		Date: peakDay, LibidoPeak: true,
	})
	user := predictorTestUser()
	stats := BuildCycleStatsFromLogs(user, logs, now, location)

	lastStart := AddCalendarDays(today, -10, location)
	nextOnset := CalendarDayKey(AddCalendarDays(lastStart, 30, location))
	ovulation := CalendarDayKey(AddCalendarDays(lastStart, 16, location))
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var envelope struct {
			Tensor struct {
				Version string `json:"version"`
				Days    []struct {
					Date        string `json:"date"`
					Peak        *int   `json:"peak"`
					PeakMissing *int   `json:"peak_missing"`
				} `json:"days"`
			} `json:"tensor"`
		}
		if err := json.NewDecoder(request.Body).Decode(&envelope); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if envelope.Tensor.Version != "v2" {
			t.Errorf("tensor version = %q, want v2", envelope.Tensor.Version)
		}
		if len(envelope.Tensor.Days) != PredictorTensorWindowDays {
			t.Errorf("tensor days = %d, want %d", len(envelope.Tensor.Days), PredictorTensorWindowDays)
		}
		sawPeak := false
		for _, row := range envelope.Tensor.Days {
			if row.Peak == nil || row.PeakMissing == nil {
				t.Errorf("day %s missing peak columns", row.Date)
				break
			}
			if *row.Peak == 1 {
				sawPeak = true
				if *row.PeakMissing != 0 {
					t.Errorf("checked peak day must be (1,0), got missing=%d", *row.PeakMissing)
				}
			}
		}
		if !sawPeak {
			t.Error("no peak=1 row reached the sidecar for a libido-tracking owner")
		}
		_, _ = writer.Write([]byte(predictorTestReply(nextOnset, ovulation, ovulation, ovulation)))
	}))
	defer server.Close()

	client := NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: server.URL, Token: "test-token", TimeoutMs: 800, TensorVersion: "v2"})
	if client == nil {
		t.Fatal("enabled v2 client is nil")
	}
	service := NewStatsService(nil, nil)
	service.ConfigurePredictor(client)
	overlaid := BuildCycleStatsFromLogs(user, logs, now, location)
	service.ApplyPredictorOverlay(context.Background(), user, logs, &overlaid, now, location)
	if !overlaid.PredictorApplied {
		t.Fatal("v2 neural overlay not applied")
	}
	if got := CalendarDayKey(overlaid.NextPeriodStart); got != nextOnset {
		t.Fatalf("v2 neural next = %s, want %s", got, nextOnset)
	}
	if overlaid.PredictorModelVersion != "ttm-r2.1-test" {
		t.Fatalf("model version not carried: %q", overlaid.PredictorModelVersion)
	}
	_ = stats
}

func TestPredictorV1OverlayUnaffectedByPeakColumns(t *testing.T) {
	// A v1-configured client keeps sending Version v1 even when the owner
	// tracks libido: existing stub/v1 stacks see zero behavior change.
	now, location := predictorTestNow()
	today := DateAtLocation(now, location)
	peakDay := AddCalendarDays(today, -20, location)
	logs := append(predictorTestHistory(today), models.DailyLog{
		Date: peakDay, LibidoPeak: true,
	})
	user := predictorTestUser()

	nextOnset := CalendarDayKey(AddCalendarDays(AddCalendarDays(today, -10, location), 30, location))
	ovulation := CalendarDayKey(AddCalendarDays(AddCalendarDays(today, -10, location), 16, location))
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var envelope struct {
			Tensor struct {
				Version string `json:"version"`
			} `json:"tensor"`
		}
		if err := json.NewDecoder(request.Body).Decode(&envelope); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if envelope.Tensor.Version != "v1" {
			t.Errorf("default client tensor version = %q, want v1", envelope.Tensor.Version)
		}
		_, _ = writer.Write([]byte(predictorTestReply(nextOnset, ovulation, ovulation, ovulation)))
	}))
	defer server.Close()

	service := NewStatsService(nil, nil)
	service.ConfigurePredictor(NewPredictorClient(PredictorConfig{Enabled: true, BaseURL: server.URL, TimeoutMs: 800}))
	overlaid := BuildCycleStatsFromLogs(user, logs, now, location)
	service.ApplyPredictorOverlay(context.Background(), user, logs, &overlaid, now, location)
	if !overlaid.PredictorApplied {
		t.Fatal("v1 overlay must still apply for a libido-tracking owner")
	}
	if got := CalendarDayKey(overlaid.NextPeriodStart); got != nextOnset {
		t.Fatalf("v1 neural next = %s, want %s", got, nextOnset)
	}
}
