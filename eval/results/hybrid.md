# Evaluation: hybrid

Agent `openai/gpt-oss-20b` · judge `openai/gpt-oss-120b` · hybrid retrieval: True · 17 cases

| Group | n | Correct | Faithful | Retrieval recall | Cites gold | Citation precision | Routing |
|---|---|---|---|---|---|---|---|
| **overall** | 17 | 100% | 94% | 100% | 92% | 100% | 100% |
| direct | 2 | 100% | – | – | – | – | 100% |
| fact | 6 | 100% | 100% | 100% | 100% | 100% | 100% |
| multi-hop | 3 | 100% | 100% | 100% | 100% | 100% | 100% |
| paraphrase | 4 | 100% | 88% | 100% | 75% | 100% | 100% |
| unanswerable | 2 | 100% | 83% | – | – | 100% | 100% |

Latency p50 13.2 s · p95 17.0 s · avg 2550 tokens per question

## Cases

| Case | Correct | Faithful | Recall | Answer |
|---|---|---|---|---|
| nw-revenue | 100% | 100% | 100% | Northwind Robotics reported total revenue of **$612 million USD** for 2025. [northwind_annual_report_2025.pdf p.1] |
| nw-segment | 100% | 100% | 100% | The largest business segment for Northwind in 2025 is **Warehouse automation**, which accounted for **58 % of total revenue**[northwind_annual_report_2025.pdf p… |
| nw-risk | 100% | 100% | 100% | Northwind’s main supply‑chain risk is its heavy dependence on a single lidar supplier, Veltrix, which provides roughly 70 % of the company’s sensors.    Sources… |
| helios-density | 100% | 100% | 100% | The Helios S2 sodium‑ion cell has an energy density of **165 Wh / kg** at the cell level.   [helios_battery_whitepaper.md] |
| helios-charge | 100% | 100% | 100% | The Helios S2 sodium‑ion cell can be charged from 10 % to 80 % state of charge in **18 minutes at 25 °C**[helios_battery_whitepaper.md]. |
| policy-days | 100% | 100% | 100% | Employees may work remotely **up to 3 days per week** under the remote work policy.   [remote_work_policy.txt] |
| nw-workforce | 100% | 100% | 100% | Northwind’s workforce grew by **360 employees** in 2025.  - Headcount at year‑end 2025: 2,140 employees[northwind_annual_report_2025.pdf p.3]   - Headcount at y… |
| helios-lifespan | 100% | 50% | 100% | The Helios S2 sodium‑ion cell is rated for **about 4,000 full charge cycles** before its capacity drops to 80 % of its original value, which is typically consid… |
| policy-abroad | 100% | 100% | 100% | Yes. According to the Remote Work Policy, employees may work from another country for **up to 30 days per calendar year**. However, you must obtain **prior writ… |
| policy-money | 100% | 100% | 100% | Remote workers receive the following financial support for setting up a home office:  - A one‑time home‑office stipend of **$600 USD**.   - A monthly contributi… |
| nw-rnd | 100% | 100% | 100% | Northwind’s 2025 annual report lists research and development spending at **$94 million USD**.   That amount represents **about 15 % of the company’s 2025 reven… |
| nw-guidance | 100% | 100% | 100% | Northwind’s 2025 annual report states that revenue for 2025 was **$612 million**[northwind_annual_report_2025.pdf p.1].   In the outlook section the company pro… |
| helios-cost | 100% | 100% | 100% | Helios plans to reduce the S2 cell’s cost from **$78 USD per kWh to $62 USD per kWh**, a cut of **$16 USD per kWh**. This target is expected to be reached by **… |
| nw-2019 | 100% | 67% | – | I couldn’t locate any information on Northwind Robotics’ 2019 revenue in the documents you provided. The only annual report available is the 2025 report, which … |
| policy-pets | 100% | 100% | – | The remote work policy does not mention pets at all. It covers eligibility, schedule, home‑office support, equipment, and working abroad, but there is no refere… |
| math | 100% | – | – | 15 percent of 200 is 30. |
| greeting | 100% | – | – | Hello! I'm here to help—just let me know what you need. |

## Issues

Incorrect answers, and correct answers containing a claim the context doesn't support.

- **helios-lifespan**: Matches reference exactly, includes the key fact of 4,000 full charge cycles to 80% capacity and adds no contradictory information.
  - unsupported claim: 80% of the original capacity is typically considered the point at which noticeable degradation begins.
- **nw-2019**: The answer correctly states that the 2019 revenue information is not available in the provided documents.
  - unsupported claim: The only annual report available is the 2025 report.

*Latency is inflated by Groq free-tier throttling: the agent and the judge share one per-minute
token quota. Unthrottled answers in this run took about 2 s.*
