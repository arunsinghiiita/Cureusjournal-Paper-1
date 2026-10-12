Appendix A. Full Source Code - edge_ai_iot_system.py

The following is the complete, unabridged source code of the reference implementation described in Section 5, provided in full to support exact reproducibility of the results reported in Section 7.

"""
edge_ai_iot_system.py
======================
Reference implementation accompanying the paper:
"A Layered Embedded Systems Architecture for Edge AI-Enabled IoT Devices:
 Design, Implementation, and Performance Evaluation"
 
This module implements a lightweight, dependency-minimal simulation of the
proposed four-layer embedded architecture (Perception -> Edge Intelligence ->
Connectivity/Middleware -> Application/Cloud) on a single host. It is written
so that the *control flow, module boundaries, and data structures* mirror an
actual MCU/SoC deployment (e.g., ESP32-S3, Arduino Nano 33 BLE Sense,
Raspberry Pi 4, NVIDIA Jetson Nano), while remaining runnable on a laptop for
reproducible experimentation.
 
Modules
-------
1. SensorNode            - Perception Layer: simulated multi-sensor stream
2. QuantizedNeuralNet     - Edge Intelligence Layer: INT8/FP32 MLP inference
3. EdgeInferenceEngine    - orchestrates preprocessing + inference + timing
4. ResourceMonitor        - tracks latency, memory footprint, energy proxy
5. MessageBroker          - Connectivity Layer: lightweight pub/sub (MQTT-like)
6. CloudSync              - Application Layer: batches + forwards edge results
7. EdgeIoTNode            - top-level orchestrator wiring all layers together
 
Author: (generated reference implementation for the accompanying journal paper)
"""
 
from __future__ import annotations
 
import json
import math
import random
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable
 
random.seed(42)
 
# --------------------------------------------------------------------------
# 1. PERCEPTION LAYER
# --------------------------------------------------------------------------
 
@dataclass

class SensorReading:
    timestamp: float
    temperature: float
    humidity: float
    vibration: float
    current: float
    label: int  # 0 = normal, 1 = anomalous (ground truth, for evaluation only)
 
 
class SensorNode:
    """Simulates a 4-channel industrial IoT sensor array (temp, humidity,
    vibration, current draw) sampled at a fixed rate, as would be produced by
    an analog front-end + ADC on a microcontroller."""
 
    def __init__(self, sample_rate_hz: float = 10.0, anomaly_prob: float = 0.12):
        self.sample_rate_hz = sample_rate_hz
        self.anomaly_prob = anomaly_prob
        self._t = 0.0
 
    def read(self) -> SensorReading:
        self._t += 1.0 / self.sample_rate_hz
        anomalous = random.random() < self.anomaly_prob
 
        base_temp = 42.0 + 3.0 * math.sin(self._t / 5.0)
        base_hum = 55.0 + 2.0 * math.sin(self._t / 7.0 + 1.0)
        base_vib = 0.15 + 0.02 * math.sin(self._t / 2.0)
        base_cur = 1.8 + 0.1 * math.sin(self._t / 3.0)
 
        if anomalous:
            base_temp += random.uniform(8, 20)
            base_vib += random.uniform(0.3, 0.9)
            base_cur += random.uniform(0.8, 2.0)
 
        noise = lambda s: random.gauss(0, s)
        reading = SensorReading(
            timestamp=self._t,
            temperature=base_temp + noise(0.4),
            humidity=base_hum + noise(0.5),
            vibration=max(0.0, base_vib + noise(0.02)),
            current=max(0.0, base_cur + noise(0.05)),
            label=int(anomalous),
        )
        return reading
 
    def stream(self, n: int) -> List[SensorReading]:

        return [self.read() for _ in range(n)]
