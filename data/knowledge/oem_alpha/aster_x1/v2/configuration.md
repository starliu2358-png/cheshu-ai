---
document_id: oem_alpha_aster_x1_v2_configuration
document_version: "2.0"
content_type: configuration
status: published
locale: zh-CN
oem_id: OEM_ALPHA
model_id: ASTER_X1
software_version: "2.0"
supersedes: oem_alpha_aster_x1_v1_configuration@1.0
content_origin: synthetic_original
---

# ASTER_X1 v2 Configuration

> 此配置表仅用于自动化评估，所有品牌、参数和功能组合均为虚构。

| Item | ASTER_X1 v2 value |
| --- | --- |
| LCC | Supported; 0–130 km/h |
| ACC | Supported; 20–150 km/h; 4 distance levels |
| AEB | Vehicle 5–90 km/h; pedestrian 5–60 km/h |
| HPA | Supported; 2 routes; 100 m per route; learning below 8 km/h |
| Seat ventilation | MAX trim only; driver seat; levels 1–2 plus Auto mode |
| Seat heating | MAX trim; driver and passenger; levels 1–3 plus Auto mode below 10°C |
| HVAC | Single-zone; 17°C–29°C; one daily preconditioning schedule |
| Battery | 75 kWh LFP; AC 7 kW; DC 180 kW; navigation-triggered preheating |
| Charge limit | 40%–100%; 5% steps; default 80% |

## Version discriminator

For ASTER_X1, the combination “LCC 0–130 km/h + HPA supports two 100 m routes + scheduled HVAC + four ACC distance levels” identifies software v2. The 75 kWh battery and MAX-only ventilation availability remain unchanged from v1; maximum DC charging increases to 180 kW.
