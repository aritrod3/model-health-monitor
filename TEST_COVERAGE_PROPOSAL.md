# Test Coverage Proposal

This document outlines proposed test coverage for the Model Health Monitor system, focusing on areas that currently lack comprehensive testing.

## Current Test Coverage

### Existing Tests
- ✅ Basic drift detection (`test_monitoring.py`)
- ✅ Worker functionality (`test_worker.py`)
- ✅ API endpoints (`test_api.py`)
- ✅ Database operations (`test_database.py`)
- ✅ Training pipeline (`test_training.py`)

## Proposed New Test Coverage

### 1. Event Queue and Async Architecture (`tests/test_events.py`)

**Purpose**: Test event queue functionality and async communication

**Test Cases**:
```python
def test_event_queue_enqueue_dequeue()
    """Test basic enqueue/dequeue operations"""
    
def test_event_queue_batch_operations()
    """Test batch dequeue operations"""
    
def test_event_queue_thread_safety()
    """Test concurrent access to event queue"""
    
def test_event_queue_maxsize_enforcement()
    """Test queue size limits and full queue behavior"""
    
def test_prediction_event_serialization()
    """Test PredictionEvent dataclass serialization"""
    
def test_reset_event_queue()
    """Test global queue reset functionality"""
```

### 2. Circuit Breaker (`tests/test_circuit_breaker.py`)

**Purpose**: Test circuit breaker logic and trust boundary enforcement

**Test Cases**:
```python
def test_circuit_breaker_no_drift()
    """Test circuit breaker allows serving when no drift"""
    
def test_circuit_breaker_critical_drift()
    """Test circuit breaker blocks serving on critical drift"""
    
def test_circuit_breaker_lookback_window()
    """Test circuit breaker respects lookback time window"""
    
def test_circuit_breaker_severity_threshold()
    """Test circuit breaker uses correct severity threshold"""
    
def test_circuit_breaker_clears_after_drift_resolves()
    """Test circuit breaker resets when drift clears"""
    
def test_circuit_breaker_integration_with_api()
    """Test circuit breaker integration with /predict endpoint"""
```

### 3. In-Process Event Processor (`tests/test_inprocess_processor.py`)

**Purpose**: Test fallback in-process event processor

**Test Cases**:
```python
def test_inprocess_processor_initialization()
    """Test processor starts when no external worker detected"""
    
def test_inprocess_processor_processes_events()
    """Test processor consumes events from queue"""
    
def test_inprocess_processor_drift_detection()
    """Test processor runs drift detection at threshold"""
    
def test_inprocess_processor_metrics_computation()
    """Test processor computes live metrics from labeled predictions"""
    
def test_inprocess_processor_graceful_shutdown()
    """Test processor stops cleanly"""
    
def test_inprocess_processor_handles_errors()
    """Test processor handles errors gracefully"""
```

### 4. Dashboard API Endpoints (`tests/test_dashboard_api.py`)

**Purpose**: Test dashboard-specific endpoints

**Test Cases**:
```python
def test_dashboard_stats_endpoint()
    """Test /api/dashboard/stats returns correct data"""
    
def test_dashboard_drift_endpoint()
    """Test /api/dashboard/drift returns recent drift signals"""
    
def test_dashboard_reset_endpoint()
    """Test /api/dashboard/reset clears database and queue"""
    
def test_dashboard_undo_stream_endpoint()
    """Test /api/dashboard/undo-stream removes predictions"""
    
def test_dashboard_stats_includes_queue_size()
    """Test stats endpoint includes event queue size"""
    
def test_dashboard_reset_clears_event_queue()
    """Test reset endpoint clears event queue"""
```

### 5. Live Metrics Computation (`tests/test_live_metrics.py`)

**Purpose**: Test rolling window metrics computation

**Test Cases**:
```python
def test_compute_metrics_from_labeled_predictions()
    """Test metrics computation from labeled predictions"""
    
def test_live_metrics_rolling_window()
    """Test rolling window limits predictions used"""
    
def test_live_metrics_with_insufficient_data()
    """Test metrics return None when insufficient labeled predictions"""
    
def test_live_metrics_updates_on_new_predictions()
    """Test metrics update as new labeled predictions arrive"""
    
def test_live_metrics_handles_missing_labels()
    """Test metrics only use predictions with labels"""
    
def test_live_metrics_roc_auc_computation()
    """Test ROC-AUC computation from probabilities"""
```

### 6. Drift Detection Integration (`tests/test_drift_integration.py`)

**Purpose**: Test end-to-end drift detection workflow

**Test Cases**:
```python
def test_drift_detection_runs_at_threshold()
    """Test drift detection runs after N predictions"""
    
def test_drift_detection_stores_signals()
    """Test drift signals are stored in database"""
    
def test_drift_detection_generates_alerts()
    """Test drift detection generates alerts for drifted features"""
    
def test_drift_detection_uses_reference_data()
    """Test drift detection compares to training reference data"""
    
def test_drift_detection_handles_insufficient_data()
    """Test drift detection handles insufficient production data"""
    
def test_drift_detection_prediction_thresholded()
    """Test drift detection is prediction-thresholded, not time-based"""
```

### 7. Error Handling and Resilience (`tests/test_resilience.py`)

**Purpose**: Test system resilience under failure conditions

**Test Cases**:
```python
def test_worker_handles_database_errors()
    """Test worker retries on database errors"""
    
def test_worker_handles_queue_full()
    """Test worker handles full event queue gracefully"""
    
def test_worker_exponential_backoff()
    """Test retry logic with exponential backoff"""
    
def test_serving_continues_when_monitoring_fails()
    """Test serving plane unaffected by monitoring failures"""
    
def test_event_queue_drops_events_when_full()
    """Test graceful degradation when queue is full"""
    
def test_system_handles_missing_model()
    """Test system handles missing model file gracefully"""
```

### 8. Configuration and Startup (`tests/test_startup.py`)

**Purpose**: Test system initialization and configuration

**Test Cases**:
```python
def test_startup_resets_database()
    """Test database reset on startup"""
    
def test_startup_clears_event_queue()
    """Test event queue cleared on startup"""
    
def test_startup_loads_model()
    """Test model loading on startup"""
    
def test_startup_starts_inprocess_processor()
    """Test in-process processor starts if no external worker"""
    
def test_startup_handles_missing_config()
    """Test startup handles missing configuration gracefully"""
    
def test_shutdown_clears_data()
    """Test shutdown event clears database"""
```

### 9. Dashboard Frontend Integration (`tests/test_dashboard_frontend.py`)

**Purpose**: Test dashboard JavaScript functionality (if using browser automation)

**Test Cases**:
```python
def test_dashboard_loads()
    """Test dashboard page loads successfully"""
    
def test_dashboard_updates_metrics()
    """Test dashboard updates metrics on load"""
    
def test_dashboard_traffic_generation()
    """Test manual traffic generation"""
    
def test_dashboard_auto_stream()
    """Test auto-stream functionality"""
    
def test_dashboard_reset_functionality()
    """Test reset button clears data"""
```

### 10. Performance and Load Testing (`tests/test_performance.py`)

**Purpose**: Test system performance under load

**Test Cases**:
```python
def test_high_throughput_prediction_handling()
    """Test system handles high prediction rate"""
    
def test_event_queue_performance()
    """Test event queue performance under load"""
    
def test_worker_batch_processing_performance()
    """Test worker batch processing efficiency"""
    
def test_database_write_performance()
    """Test database write performance"""
    
def test_drift_detection_performance()
    """Test drift detection performance with large datasets"""
```

## Priority Order

1. **High Priority** (Core functionality):
   - Circuit breaker tests
   - Dashboard API endpoints
   - Live metrics computation
   - Drift detection integration

2. **Medium Priority** (Important features):
   - Event queue tests
   - In-process processor tests
   - Error handling and resilience

3. **Low Priority** (Nice to have):
   - Frontend integration tests
   - Performance tests
   - Configuration tests

## Implementation Notes

- Use `pytest` fixtures for common setup (database, event queue, etc.)
- Mock external dependencies (file system, network) where appropriate
- Use `unittest.mock` for testing async behavior
- Consider using `pytest-asyncio` for async test support
- Add integration tests that test full workflows end-to-end

## Coverage Goals

- **Target**: 80%+ code coverage
- **Focus**: Critical paths (circuit breaker, drift detection, metrics computation)
- **Exclude**: Generated code, configuration files, test utilities
