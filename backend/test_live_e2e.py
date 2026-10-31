import httpx
import sys

def run_live_e2e_test():
    base_url = "http://127.0.0.1:8000"

    print("=" * 60)
    print("LIVE E2E PIPELINE TEST: PHASES 1 -> 2 -> 3 -> 4 -> 5")
    print("=" * 60)

    # 1. Health check
    r = httpx.get(f"{base_url}/")
    print(f"1. Root Endpoint: {r.status_code} -> {r.json()}")
    assert r.status_code == 200
    assert r.json()["phase"] == 7

    # 2. Phase 1 Profile
    csv_data = "salary,experience,education,age\n50000,2,Bachelor,24\n60000,3,Master,27\n75000,5,Master,30\n110000,8,PhD,35\n120000,10,PhD,40\n52000,2,Bachelor,25\n58000,3,Bachelor,26\n72000,4,Master,29\n105000,7,PhD,34\n130000,12,PhD,42\n"
    files = {"file": ("test_dataset.csv", csv_data.encode("utf-8"), "text/csv")}
    r_profile = httpx.post(f"{base_url}/api/profile", files=files)
    print(f"2. Phase 1 Profile: {r_profile.status_code}")
    assert r_profile.status_code == 200
    profile = r_profile.json()
    dataset_id = profile["dataset_id"]
    print(f"   Dataset ID: {dataset_id}, Rows: {profile['overview']['n_rows']}, Cols: {profile['overview']['n_cols']}")

    # 3. Phase 2 Plan
    plan_payload = {
        "profile": profile,
        "objective": "Investigate what factors influence salary",
        "dataset_id": dataset_id,
    }
    r_plan = httpx.post(f"{base_url}/api/plan", json=plan_payload)
    print(f"3. Phase 2 Plan: {r_plan.status_code}")
    assert r_plan.status_code == 200
    plan = r_plan.json()
    print(f"   Tasks planned: {len(plan['analysis_tasks'])}")

    # 4. Phase 3 Analysis Execution
    analysis_payload = {
        "dataset_id": dataset_id,
        "tasks": plan["analysis_tasks"],
    }
    r_analysis = httpx.post(f"{base_url}/api/analyze-plan", json=analysis_payload)
    print(f"4. Phase 3 Analysis: {r_analysis.status_code}")
    assert r_analysis.status_code == 200
    analysis = r_analysis.json()
    print(f"   Results computed: {len(analysis['results'])}, Successful: {analysis['successful_tasks']}")

    # 5. Phase 4 Critic Review
    critic_payload = {
        "analysis_response": analysis,
        "dataset_profile": profile,
        "investigation_plan": plan,
        "dataset_id": dataset_id,
    }
    r_critic = httpx.post(f"{base_url}/api/critic", json=critic_payload)
    print(f"5. Phase 4 Critic Review: {r_critic.status_code}")
    assert r_critic.status_code == 200
    critic = r_critic.json()
    print(f"   Reviews generated: {len(critic['reviews'])}, Accepted: {critic['summary']['accepted']}, Re-investigate: {critic['summary']['requires_reinvestigation']}")

    # 6. Phase 5 Verification & Re-investigation
    verification_payload = {
        "analysis_response": analysis,
        "critic_response": critic,
        "dataset_profile": profile,
        "investigation_plan": plan,
        "dataset_id": dataset_id,
        "max_reinvestigations": 1,
    }
    r_verify = httpx.post(f"{base_url}/api/verify-findings", json=verification_payload)
    print(f"6. Phase 5 Verified Findings: {r_verify.status_code}")
    assert r_verify.status_code == 200
    verified = r_verify.json()
    print(f"   Summary: {verified['summary']}")
    print(f"   Verified findings count: {len(verified['verified_findings'])}")

    # 7. Phase 6 Insight Generation
    insights_payload = {
        "verification_response": verified,
        "dataset_profile": profile,
        "investigation_plan": plan,
        "dataset_id": dataset_id,
    }
    r_insights = httpx.post(f"{base_url}/api/insights", json=insights_payload)
    print(f"7. Phase 6 Key Insights: {r_insights.status_code}")
    assert r_insights.status_code == 200
    insights = r_insights.json()
    print(f"   Total Insights Generated: {insights['summary']['total_insights']}")
    print(f"   Strong: {insights['summary']['strong_insights']}, Moderate: {insights['summary']['moderate_insights']}, Weak: {insights['summary']['weak_insights']}")
    if insights['key_insights']:
        top_ins = insights['key_insights'][0]
        print(f"   Top Insight: '{top_ins['title']}' ({top_ins['strength']} strength)")
        print(f"   Evidence: Method={top_ins['evidence']['method']}, p={top_ins['evidence']['p_value']}")


    # 8. Phase 7 System Evaluation
    evaluation_payload = {
        "profile": profile,
        "plan": plan,
        "analysis": analysis,
        "critic": critic,
        "verification": verified,
        "insights": insights,
    }
    r_eval = httpx.post(f"{base_url}/api/evaluate", json=evaluation_payload)
    print(f"8. Phase 7 System Evaluation: {r_eval.status_code}")
    assert r_eval.status_code == 200
    evaluation = r_eval.json()
    print(f"   Overall Score: {evaluation['summary']['overall_score']}/100 ({evaluation['summary']['overall_grade']})")
    print(f"   Zero-Hallucination: {evaluation['summary']['zero_hallucination_verified']}")
    for dim in evaluation['dimensions']:
        print(f"   - {dim['name']}: {dim['score']}/{dim['max_score']} ({dim['percentage']}%, {dim['grade']})")

    # 9. Test targeted re-investigation with extreme outlier data (triggers Pearson -> Spearman)
    print("\n9. Testing Targeted Re-investigation Pipeline...")
    skewed_csv = "x,y\n1,2\n2,4\n3,6\n4,8\n5,10\n6,12\n7,14\n8,16\n9,18\n1000,20\n"
    r_prof2 = httpx.post(f"{base_url}/api/profile", files={"file": ("skewed.csv", skewed_csv.encode("utf-8"), "text/csv")})
    assert r_prof2.status_code == 200
    prof2 = r_prof2.json()
    
    r_plan2 = httpx.post(f"{base_url}/api/plan", json={"profile": prof2, "objective": "Analyze x vs y", "dataset_id": prof2["dataset_id"]})
    assert r_plan2.status_code == 200
    plan2 = r_plan2.json()

    r_ana2 = httpx.post(f"{base_url}/api/analyze-plan", json={"dataset_id": prof2["dataset_id"], "tasks": plan2["analysis_tasks"]})
    assert r_ana2.status_code == 200
    ana2 = r_ana2.json()

    r_crit2 = httpx.post(f"{base_url}/api/critic", json={"analysis_response": ana2, "dataset_profile": prof2, "investigation_plan": plan2, "dataset_id": prof2["dataset_id"]})
    assert r_crit2.status_code == 200
    crit2 = r_crit2.json()
    print(f"   Critic Reviews: Accepted={crit2['summary']['accepted']}, Re-investigate={crit2['summary']['requires_reinvestigation']}")

    r_ver2 = httpx.post(f"{base_url}/api/verify-findings", json={"analysis_response": ana2, "critic_response": crit2, "dataset_profile": prof2, "investigation_plan": plan2, "dataset_id": prof2["dataset_id"]})
    assert r_ver2.status_code == 200
    ver2 = r_ver2.json()
    print(f"   Phase 5 Summary: Verified={ver2['summary']['verified']}, Reinvestigated-Verified={ver2['summary']['verified_after_reinvestigation']}, Total Reinvestigated={ver2['summary']['total_reinvestigated']}")
    print(f"   Audit trail count: {len(ver2['reinvestigations'])}")

    r_ins2 = httpx.post(f"{base_url}/api/insights", json={"verification_response": ver2, "dataset_profile": prof2, "investigation_plan": plan2, "dataset_id": prof2["dataset_id"]})
    assert r_ins2.status_code == 200
    ins2 = r_ins2.json()
    print(f"   Phase 6 Summary: Total Insights={ins2['summary']['total_insights']}, Strong={ins2['summary']['strong_insights']}")

    r_eval2 = httpx.post(f"{base_url}/api/evaluate", json={"profile": prof2, "plan": plan2, "analysis": ana2, "critic": crit2, "verification": ver2, "insights": ins2})
    assert r_eval2.status_code == 200
    eval2 = r_eval2.json()
    print(f"   Phase 7 Summary: Score={eval2['summary']['overall_score']}/100 ({eval2['summary']['overall_grade']})")

    print("=" * 60)
    print("ALL 7 PHASES + RE-INVESTIGATION VALIDATED SUCCESSFULLY END-TO-END!")
    print("=" * 60)



if __name__ == "__main__":
    try:
        run_live_e2e_test()
    except Exception as e:
        print(f"Error during E2E test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
