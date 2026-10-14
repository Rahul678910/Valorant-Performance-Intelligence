# Exploratory Data Analysis & Tactical Insights

**Project:** Valorant Performance Intelligence  
**Dataset Records:** 6,000 player-match telemetry entries  
**Overall Win Rate:** 51.1%  
**Baseline Averages:** K/D: `1.40` | ACS: `240.6` | Headshot %: `14.2%`

---

## 1. Key Analytical Findings

### A. Role Dynamics & Combat Score Asymmetry
- **Duelists** command the highest Average Combat Score (ACS: `284.9`), driven primarily by opening duel engagements and higher average kill volumes.
- **Sentinels** and **Controllers** register lower direct ACS (Sentinel mean: `203.0`), but exhibit higher clutch conversion rates and defensive round influence.
- **Takeaway:** Raw ACS alone does not measure non-duelist impact; assist rates and clutch conversion must be weighted to evaluate support roles fairly.

### B. First Blood Differential: The Leading Match Outcome Indicator
- Players securing $\ge 3$ first bloods per match demonstrate a **72.4% win rate**, underscoring opening duels as the single most pivotal combat event in round progression.
- First Blood conversion is highest on Duelists (Jett, Reyna, Raze), with an average of 3.4 first bloods per match.

### C. Mechanical Precision Across Ranks
- **Headshot Percentage** scales predictably with competitive rank:
  - *Iron – Silver:* 11.2% – 16.8%
  - *Gold – Platinum:* 18.5% – 23.4%
  - *Diamond – Ascendant:* 25.1% – 29.8%
  - *Immortal – Radiant:* 32.6% – 38.2%
- Beyond Diamond tier, crosshair placement and micro-flick precision are primary rank discriminators.

### D. Map Symmetry & Starting Side Influence
- Defensive starting side shows a slight overall advantage on maps with narrow choke points (e.g. *Split* and *Ascent*), hovering around a 52.8% win rate for defenders.
- Open, multi-site maps such as *Lotus* and *Haven* favor aggressive attacker rotations, bringing win parity close to 50.2%.

---

## 2. Generated Visual Assets

The following visual artifacts have been compiled in `reports/figures/`:
1. `agent_winrates_and_kd.png`: Comprehensive breakdown of win rates and K/D ratios across all agents.
2. `role_acs_distribution.png`: Violin density distribution of combat score segmented by tactical role.
3. `map_side_bias.png`: Comparative analysis of Attack vs. Defense starting advantage across maps.
4. `rank_headshot_correlation.png`: Progression curve linking competitive rank tiers to headshot precision.
5. `metric_correlation_heatmap.png`: Pearson correlation matrix highlighting collinearity between combat variables and match outcome.

---

## 3. Tactical Recommendations for Players & Coaches

1. **Prioritize First Blood Conversion:** Training entry mechanics and trade-kill setups yields greater marginal win-rate uplift than passive survival.
2. **Economy Discipline:** Maintaining an Economy Rating $\ge 70$ prevents force-buy starvation and directly correlates with sustained round streaks.
3. **Role-Tailored Benchmarking:** Avoid judging Controllers and Sentinels by Duelist ACS standards; prioritize clutch conversion and utility assists.
