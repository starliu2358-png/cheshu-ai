---
document_id: oem_alpha_aster_x1_v1_configuration
document_version: "1.0"
content_type: configuration
status: published
locale: zh-CN
oem_id: OEM_ALPHA
model_id: ASTER_X1
software_version: "1.0"
content_origin: synthetic_original
---

# ASTER_X1 v1 Configuration

> 此配置表仅用于自动化评估，所有品牌、参数和功能组合均为虚构。

| Item | ASTER_X1 v1 value |
| --- | --- |
| LCC | Supported; 60–120 km/h |
| ACC | Supported; 30–150 km/h; 3 distance levels |
| AEB | Vehicle 8–80 km/h; pedestrian 8–50 km/h |
| HPA | Not supported |
| Seat ventilation | MAX trim only; driver seat; levels 1–2; no Auto mode |
| Seat heating | MAX trim; driver and passenger; levels 1–3; no Auto mode |
| HVAC | Single-zone; 17°C–29°C; no scheduled preconditioning |
| Battery | 75 kWh LFP; AC 7 kW; DC 150 kW |
| Charge limit | 50%–100%; 10% steps; default 90% |

## Version discriminator

For ASTER_X1, the combination “LCC 60–120 km/h + HPA unavailable + no scheduled HVAC + three ACC distance levels” identifies software v1. Do not use ASTER_X1 v2 menu paths or limits for this version.
