---
document_id: oem_alpha_aster_x1_v2_adas
document_version: "2.0"
content_type: adas
status: published
locale: zh-CN
oem_id: OEM_ALPHA
model_id: ASTER_X1
software_version: "2.0"
supersedes: oem_alpha_aster_x1_v1_adas@1.0
content_origin: synthetic_original
---

# ASTER_X1 v2 ADAS Guide

> 以下功能均为驾驶辅助，不替代驾驶员观察、判断或控制。本文内容完全虚构。

## Lane Centering Control

Feature: LCC

LCC can be activated when:

- vehicle speed is between 0 km/h and 130 km/h
- lane environment satisfies system requirements
- driver supervision is available

Press the lane button on the left side of the steering wheel to activate LCC. A blue steering-wheel icon means LCC is active; a gray icon means the activation conditions are not satisfied.

## ACC

自适应巡航（ACC）的可设定车速范围为 20–150 km/h，跟车距离提供 1–4 共 4 档。v2 支持跟停后 10 秒内自动起步；停车超过 10 秒后，需按“恢复”键或轻踩加速踏板。

## AEB

自动紧急制动（AEB）可在 5–90 km/h 范围内对前方车辆风险提供制动辅助，并在 5–60 km/h 范围内识别行人风险。每次启动车辆时 AEB 默认开启；驾驶员不能通过语音永久关闭 AEB。

## HPA

HPA is supported on ASTER X1 software version 2.0. The vehicle can save up to 2 routes, each no longer than 100 m. Route learning requires a speed below 8 km/h. During playback, select a route that matches the current location and keep the on-screen HPA Start button pressed.
