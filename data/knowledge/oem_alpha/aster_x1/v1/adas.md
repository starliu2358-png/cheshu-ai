---
document_id: oem_alpha_aster_x1_v1_adas
document_version: "1.0"
content_type: adas
status: published
locale: zh-CN
oem_id: OEM_ALPHA
model_id: ASTER_X1
software_version: "1.0"
content_origin: synthetic_original
---

# ASTER_X1 v1 ADAS Guide

> 以下功能均为驾驶辅助，不替代驾驶员观察、判断或控制。本文内容完全虚构。

## Lane Centering Control

Feature: LCC

LCC can be activated when:

- vehicle speed is between 60 km/h and 120 km/h
- left and right lane markings are valid
- driver hands are detected on the steering wheel

Press the lane button on the left side of the steering wheel to activate LCC. A green steering-wheel icon means LCC is active; a gray icon means the activation conditions are not satisfied.

## ACC

自适应巡航（ACC）的可设定车速范围为 30–150 km/h。按方向盘左侧“巡航”键启用，跟车距离提供近、中、远 3 档。v1 不支持跟停后自动起步；停车超过 3 秒后需踩加速踏板恢复。

## AEB

自动紧急制动（AEB）可在 8–80 km/h 范围内对前方车辆风险提供制动辅助，并在 8–50 km/h 范围内识别行人风险。每次启动车辆时 AEB 默认开启。恶劣天气、摄像头遮挡或附着力不足可能限制功能。

## HPA

HPA is not available on ASTER X1 software version 1.0. The settings do not contain an HPA or memorized-route entry.
