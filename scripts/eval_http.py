#!/usr/bin/env python3
"""
scripts/eval_http.py  — fixed-seed benchmark via HTTP.

Usage:
    python scripts/eval_http.py --seeds 10 --base-url http://localhost:8000
    python scripts/eval_http.py --seeds 5 --tag trained --compare results/baseline.json
"""
import argparse, json, os, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from client.openenv_client import OpenEnvClient
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; HAS_MPL=True
except ImportError: HAS_MPL=False

AGENTS=["agent1","agent2"]; MAX_STEPS={"easy":150,"medium":200,"hard":300}

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

def run_eval(base_url,n_seeds,tasks,mode="default",program_id=None):
    client=OpenEnvClient(base_url)
    print(f"Waiting for server at {base_url}...")
    if not client.wait_for_server(): raise RuntimeError("Server not responding")
    results=defaultdict(list); intervention_counts=defaultdict(list); completion_rates=defaultdict(list)
    total=len(tasks)*n_seeds; idx=0
    for task in tasks:
        max_s=MAX_STEPS.get(task,300)
        for seed in range(n_seeds):
            idx+=1; obs=client.reset(task=task,seed=seed,mode=mode,program_id=program_id)
            done=False; step=0; ep_r=0.0; intv=0; info={}
            while not done and step<max_s:
                step+=1
                for aid in AGENTS:
                    if done: break
                    act=_heuristic(aid,obs,info); r=client.step(**act)
                    ep_r+=r.reward; done=r.done; info=r.info; obs=r.observation
                    if r.fleet_ai_intervention: intv+=1
            n_o=len(obs.get("orders",[1])); cr=info.get("completed_orders",0)/max(n_o,1)
            results[task].append(ep_r); intervention_counts[task].append(intv); completion_rates[task].append(cr)
            print(f"  [{idx}/{total}] task={task} seed={seed} reward={ep_r:.3f} comp={cr:.2f} intv={intv}")
    summary={"mode":mode,"program_id":program_id,"n_seeds":n_seeds}
    for task in tasks:
        rws=results[task]; mean_r=sum(rws)/max(len(rws),1)
        std_r=(sum((r-mean_r)**2 for r in rws)/max(len(rws),1))**0.5
        crs=completion_rates[task]
        summary[task]={"mean_reward":round(mean_r,4),"std_reward":round(std_r,4),
                       "mean_completion":round(sum(crs)/max(len(crs),1),3),
                       "mean_interventions":round(sum(intervention_counts[task])/max(len(intervention_counts[task]),1),1),
                       "rewards":[round(r,4) for r in rws]}
    return summary

def plot_results(baseline,trained,out_dir):
    if not HAS_MPL: print("matplotlib not available — skip plots"); return
    os.makedirs(out_dir,exist_ok=True)
    tasks=["easy","medium","hard"]; fig,axes=plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Smart Warehouse — Benchmark Results",fontsize=15,fontweight="bold")
    colors=["#6366F1","#10B981"]
    for i,task in enumerate(tasks):
        ax=axes[i]; b=baseline.get(task,{}); t_=trained.get(task,{}) if trained else {}
        bars=[b.get("mean_reward",0)]; errs=[b.get("std_reward",0)]; labels=["Baseline"]; cs=[colors[0]]
        if t_: bars.append(t_.get("mean_reward",0)); errs.append(t_.get("std_reward",0)); labels.append("Trained"); cs.append(colors[1])
        ax.bar(range(len(bars)),bars,yerr=errs,color=cs,alpha=0.85,width=0.5,capsize=5)
        ax.set_xticks(range(len(bars))); ax.set_xticklabels(labels,fontsize=11)
        ax.set_title(f"Task: {task}",fontsize=13); ax.set_ylabel("Mean Reward",fontsize=11); ax.grid(True,alpha=0.3,axis="y")
        for xi,(b_,e_) in enumerate(zip(bars,errs)): ax.text(xi,b_+e_+0.02,f"{b_:.3f}",ha="center",fontsize=9,fontweight="bold")
    plt.tight_layout(); path=os.path.join(out_dir,"benchmark_results.png"); plt.savefig(path,dpi=150); plt.close(); print(f"Plot: {path}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base-url",default="http://localhost:8000")
    ap.add_argument("--seeds",type=int,default=10); ap.add_argument("--tasks",nargs="+",default=["easy","medium","hard"])
    ap.add_argument("--mode",default="default"); ap.add_argument("--program-id",default=None)
    ap.add_argument("--tag",default="baseline"); ap.add_argument("--compare",default=None); ap.add_argument("--out-dir",default="results")
    args=ap.parse_args()
    summary=run_eval(args.base_url,args.seeds,args.tasks,args.mode,args.program_id)
    os.makedirs(args.out_dir,exist_ok=True); out=os.path.join(args.out_dir,f"{args.tag}.json")
    with open(out,"w") as f: json.dump(summary,f,indent=2)
    print(f"\nSaved: {out}")
    compare=None
    if args.compare and os.path.exists(args.compare):
        with open(args.compare) as f: compare=json.load(f)
    plot_results(compare or summary, summary if compare else None, os.path.join(args.out_dir,"plots"))

if __name__=="__main__": main()
