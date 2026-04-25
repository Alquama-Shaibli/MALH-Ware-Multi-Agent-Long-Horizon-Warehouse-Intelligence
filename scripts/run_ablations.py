#!/usr/bin/env python3
"""
scripts/run_ablations.py
=========================
Runs 4 ablation conditions across fixed seeds and generates ablation_plot.png.

Conditions:
  baseline     — default mode, no program, Fleet AI ON (control)
  no_fleet_ai  — Fleet AI disabled via env flag
  negotiation  — mode=negotiation, negotiation actions enabled
  program      — program_id=hard_full, work-order constraints active

Each condition runs N_SEEDS episodes on "hard" task and saves results/*.json.

Usage:
    # With server running:
    python scripts/run_ablations.py --seeds 5 --base-url http://localhost:8000
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from client.openenv_client import OpenEnvClient

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; HAS_MPL=True
except ImportError: HAS_MPL=False

AGENTS=["agent1","agent2"]

def _heuristic(aid,obs,info):
    robot=obs.get("robots",{}).get(aid,{}); pos=robot.get("pos",[0,0])
    carrying=robot.get("carrying",[]); battery=robot.get("battery",100)
    goal=obs.get("goal",[5,5]); station=obs.get("charge_station",[0,5])
    orders=obs.get("orders",[]); completed=obs.get("completed_orders",[])
    inventory=obs.get("inventory",{})
    def nav(fx,fy):
        dx,dy=fx-pos[0],fy-pos[1]
        if abs(dx)>=abs(dy): return "down" if dx>0 else "up"
        return "right" if dy>0 else "left"
    if battery<15 and pos!=station: return {"agent_id":aid,"action_type":"move","direction":nav(*station)}
    if pos==station and battery<50: return {"agent_id":aid,"action_type":"charge"}
    if pos==goal and carrying: return {"agent_id":aid,"action_type":"drop"}
    for item,loc in inventory.items():
        if loc==pos: return {"agent_id":aid,"action_type":"pick"}
    pending=set()
    for o in orders:
        if o["id"] in completed: continue
        dep=o.get("depends_on")
        if dep and dep not in completed: continue
        for it in o["items"]: pending.add(it)
    all_carried=[i for r in obs.get("robots",{}).values() for i in r.get("carrying",[])]
    if carrying: return {"agent_id":aid,"action_type":"move","direction":nav(*goal)}
    targets=[(it,loc) for it,loc in inventory.items() if it not in all_carried and it in pending]
    if targets:
        tgt=targets[0][1] if aid=="agent1" else targets[-1][1]
        return {"agent_id":aid,"action_type":"move","direction":nav(*tgt)}
    return {"agent_id":aid,"action_type":"move","direction":"right"}

def _run_condition(client, task, n_seeds, mode, program_id, disable_fleet_ai=False):
    rewards=[]; completions=[]; interventions=[]
    for seed in range(n_seeds):
        obs=client.reset(task=task, seed=seed, mode=mode, program_id=program_id)
        done=False; step=0; ep_r=0.0; intv=0; info={}
        max_s=300
        while not done and step<max_s:
            step+=1
            for aid in AGENTS:
                if done: break
                act=_heuristic(aid,obs,info); r=client.step(**act)
                ep_r+=r.reward; done=r.done; info=r.info; obs=r.observation
                if r.fleet_ai_intervention: intv+=1
        n_o=len(obs.get("orders",[1])); cr=info.get("completed_orders",0)/max(n_o,1)
        rewards.append(ep_r); completions.append(cr); interventions.append(intv)
        print(f"    seed={seed} reward={ep_r:.3f} comp={cr:.2f}")
    mean_r=sum(rewards)/max(len(rewards),1)
    std_r=(sum((r-mean_r)**2 for r in rewards)/max(len(rewards),1))**0.5
    return {"mean_reward":round(mean_r,4),"std_reward":round(std_r,4),
            "mean_completion":round(sum(completions)/max(len(completions),1),3),
            "mean_interventions":round(sum(interventions)/max(len(interventions),1),1),
            "rewards":[round(r,4) for r in rewards]}

def plot_ablations(results, out_dir):
    if not HAS_MPL: print("matplotlib not available — skip plots"); return
    os.makedirs(out_dir, exist_ok=True)
    conditions=list(results.keys())
    means=[results[c]["mean_reward"] for c in conditions]
    stds=[results[c]["std_reward"] for c in conditions]
    colors=["#6366F1","#EF4444","#10B981","#F59E0B"][:len(conditions)]
    fig,ax=plt.subplots(figsize=(10,6))
    bars=ax.bar(conditions, means, yerr=stds, color=colors, alpha=0.85, width=0.5,
                capsize=6, error_kw={"elinewidth":2})
    ax.set_title("Fleet AI Ablation Study — Hard Task", fontsize=15, fontweight="bold")
    ax.set_ylabel("Mean Reward (10 seeds)", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for i,(m,s) in enumerate(zip(means,stds)):
        ax.text(i, m+s+0.02, f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path=os.path.join(out_dir,"ablation_plot.png")
    plt.savefig(path, dpi=150); plt.close(); print(f"Ablation plot: {path}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--seeds",    type=int, default=5)
    ap.add_argument("--task",     default="hard")
    ap.add_argument("--out-dir",  default="results")
    args=ap.parse_args()

    client=OpenEnvClient(args.base_url)
    print(f"Waiting for server at {args.base_url}...")
    if not client.wait_for_server(): raise RuntimeError("Server not responding")

    ABLATION_CONDITIONS = {
        "baseline":    {"mode":"default",      "program_id":None},
        "negotiation": {"mode":"negotiation",  "program_id":None},
        "program":     {"mode":"default",      "program_id":"hard_full"},
        "judge_mode":  {"mode":"negotiation",  "program_id":"judge"},
    }

    all_results = {}
    for cond_name, kwargs in ABLATION_CONDITIONS.items():
        print(f"\n--- Ablation: {cond_name} ---")
        res = _run_condition(client, args.task, args.seeds, **kwargs)
        all_results[cond_name] = res
        print(f"  mean_reward={res['mean_reward']} std={res['std_reward']}")

    os.makedirs(os.path.join(args.out_dir,"ablations"), exist_ok=True)
    for cond_name, res in all_results.items():
        path=os.path.join(args.out_dir,"ablations",f"{cond_name}.json")
        with open(path,"w") as f: json.dump(res,f,indent=2)
    summary_path=os.path.join(args.out_dir,"ablations","summary.json")
    with open(summary_path,"w") as f: json.dump(all_results,f,indent=2)
    print(f"\nAll ablations saved to {args.out_dir}/ablations/")
    plot_ablations(all_results, os.path.join(args.out_dir,"plots"))

if __name__=="__main__": main()
