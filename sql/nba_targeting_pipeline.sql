-- Next Best Action targeting & measurement pipeline (BigQuery Standard SQL).
-- The SQL layer campaign ops runs alongside the Python models: audience sizing,
-- eligibility funnels, capacity checks, and post-campaign performance reporting.

-- 1. Audience eligibility funnel per action (inclusion/exclusion rules in SQL)
WITH eligibility AS (
  SELECT
    member_id,
    open_care_gaps >= 1                          AS elig_care_gap_outreach,
    age < 70                                     AS elig_wellness_program,
    digital_active = 1                           AS elig_telehealth_nudge
  FROM `nba.members`
)
SELECT
  COUNTIF(elig_care_gap_outreach) AS n_care_gap_eligible,
  COUNTIF(elig_wellness_program)  AS n_wellness_eligible,
  COUNTIF(elig_telehealth_nudge)  AS n_telehealth_eligible,
  COUNT(*)                        AS total_members
FROM eligibility;

-- 2. Campaign sizing with capacity cap: rank eligible members by expected value
--    (uplift x action_value) and take the top-N per action (QUALIFY on BigQuery)
SELECT member_id, action, expected_value
FROM (
  SELECT s.member_id, s.action, s.uplift * a.action_value AS expected_value,
         ROW_NUMBER() OVER (PARTITION BY s.action ORDER BY s.uplift * a.action_value DESC) AS rnk
  FROM `nba.uplift_scores` s
  JOIN `nba.action_values` a USING (action)
  WHERE s.uplift > 0
)
WHERE rnk <= (SELECT capacity FROM `nba.action_capacity` c WHERE c.action = action)
ORDER BY action, expected_value DESC;

-- 3. Contact-fatigue de-duplication: one action per member (the global NBA)
--    keep each member's single highest-expected-value eligible action
SELECT member_id, action AS next_best_action, expected_value
FROM (
  SELECT member_id, action, expected_value,
         ROW_NUMBER() OVER (PARTITION BY member_id ORDER BY expected_value DESC) AS pick
  FROM `nba.campaign_candidates`
)
WHERE pick = 1;

-- 4. Post-campaign performance: treatment vs control response by action
--    (the measurement framework's A/B lift, computed in-warehouse)
SELECT
  action,
  COUNTIF(treatment = 1)                                             AS n_treated,
  COUNTIF(treatment = 0)                                             AS n_control,
  ROUND(AVG(IF(treatment = 1, response, NULL)), 4)                  AS resp_treatment,
  ROUND(AVG(IF(treatment = 0, response, NULL)), 4)                  AS resp_control,
  ROUND(AVG(IF(treatment = 1, response, NULL))
      - AVG(IF(treatment = 0, response, NULL)), 4)                  AS absolute_lift
FROM `nba.experiment`
GROUP BY action
ORDER BY absolute_lift DESC;
