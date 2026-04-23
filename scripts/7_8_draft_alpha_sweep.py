#!/usr/bin/env python3
"""
Fast grid search: (n_drafts, alpha, K) on 100 GSM8K + 100 MATH500.

n_drafts in {2,4,8}, alpha in {0.3..0.7}, K in {8,16,32}.
All generation merged into ONE batch per GPU (model loaded once).
Checkpoint/resume. Prints LTE score table at the end.

LTE (Late-Token Efficiency) = gain_over_greedy / compute_multiplier
  compute_multiplier = tokens_per_q / greedy_tokens_per_q

Usage:
    python scripts/7_8_draft_alpha_sweep.py --gpus 2,3,4,5,6,7
"""

import argparse, json, math, os, re, subprocess, sys, time, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-3B-Instruct"
SYS = "Please reason step by step, and put your final answer within \\boxed{}."

def split_steps(t):
    t=(t or"").strip()
    if not t:return[]
    s=[x.strip()for x in t.split("\n\n")if x.strip()]
    return s if s else[t]

def prompt(q):
    return f"<|im_start|>system\n{SYS}\n<|im_end|>\n<|im_start|>user\n{q.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

def boxed(t):
    idx=(t or"").rfind("\\boxed")
    if idx<0:return""
    i,d,s=idx,0,None
    while i<len(t):
        if t[i]=="{":
            if d==0:s=i
            d+=1
        elif t[i]=="}":
            d-=1
            if d==0 and s is not None:return t[s+1:i].strip()
        i+=1
    return""

def norm(t):
    t=(t or"").strip().replace("$","").replace(",","")
    t=re.sub(r"\\boxed\{(.*)\}",r"\1",t)
    t=re.sub(r"\\text\{(.*?)\}",r"\1",t)
    t=re.sub(r"\\(?:frac|dfrac)\{([^}]*)\}\{([^}]*)\}",r"\1/\2",t)
    return re.sub(r"\s+","",t).lower()

def vote(answers):
    ns=[norm(a)for a in answers if norm(a)]
    return Counter(ns).most_common(1)[0][0]if ns else""

def load_jsonl(p):
    p=Path(p)
    if not p.exists():return[]
    return[json.loads(l)for l in p.read_text("utf-8").splitlines()if l.strip()]

def append_jsonl(p,recs):
    with Path(p).open("a",encoding="utf-8")as f:
        for r in recs:f.write(json.dumps(r,ensure_ascii=False)+"\n")

def load_gsm8k(n,seed=42):
    raw=str(ROOT/"results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl")
    seen,qs=set(),[]
    for line in open(raw):
        d=json.loads(line)
        if d["doc_id"]not in seen:
            seen.add(d["doc_id"])
            qs.append({"doc_id":d["doc_id"],"question":d["question"],
                       "gold_answer":d["gold_answer"],"dataset":"gsm8k"})
    random.Random(seed).shuffle(qs)
    return qs[:n]

def load_math500(n,seed=42):
    raw=str(ROOT/"lm-evaluation-harness/math_eval_data/MATH-500/test.jsonl")
    qs=[]
    for i,line in enumerate(open(raw)):
        d=json.loads(line)
        qs.append({"doc_id":10000+i,"question":d["problem"],
                   "gold_answer":d["answer"],"dataset":"math500"})
    random.Random(seed).shuffle(qs)
    return qs[:n]

def fmt(s):
    m,s=divmod(int(s),60);return f"{m}m{s:02d}s"


def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-id",default=MODEL)
    ap.add_argument("--out-dir",default=str(ROOT/"results/draft_alpha_sweep"))
    ap.add_argument("--n-gsm8k",type=int,default=100)
    ap.add_argument("--n-math500",type=int,default=100)
    ap.add_argument("--n-drafts-list",default="2,4,8")
    ap.add_argument("--alphas",default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--Ks",default="8,16,32")
    ap.add_argument("--temperature",type=float,default=0.7)
    ap.add_argument("--top-p",type=float,default=0.9)
    ap.add_argument("--max-tokens",type=int,default=1024)
    ap.add_argument("--gpus",default="2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization",type=float,default=0.90)
    ap.add_argument("--max-model-len",type=int,default=2048)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--_sid",type=int,default=-1,help=argparse.SUPPRESS)
    ap.add_argument("--_out",default="",help=argparse.SUPPRESS)
    ap.add_argument("--_gpu",default="0",help=argparse.SUPPRESS)
    ap.add_argument("--_tf",default="",help=argparse.SUPPRESS)
    return ap.parse_args()


# ── shard worker ──────────────────────────────────────────────────────────

def run_shard(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=args._gpu
    from vllm import LLM,SamplingParams

    tasks=json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} tasks")

    llm=LLM(model=args.model_id,tensor_parallel_size=1,
            trust_remote_code=True,dtype="half",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len)

    sp_g=SamplingParams(temperature=0.0,max_tokens=args.max_tokens,
                        stop=["<|im_end|>","<|endoftext|>"])
    sp_s=SamplingParams(temperature=args.temperature,top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        stop=["<|im_end|>","<|endoftext|>"])

    gi=[i for i,t in enumerate(tasks)if t.get("mode")=="greedy"]
    si=[i for i,t in enumerate(tasks)if t.get("mode")!="greedy"]

    res=[None]*len(tasks)
    if gi:
        o=llm.generate([tasks[i]["prompt"]for i in gi],sp_g)
        for j,i in enumerate(gi):res[i]=o[j]
    if si:
        o=llm.generate([tasks[i]["prompt"]for i in si],sp_s)
        for j,i in enumerate(si):res[i]=o[j]

    with Path(args._out).open("w",encoding="utf-8")as fout:
        for task,output in zip(tasks,res):
            g=output.outputs[0].text.strip()
            nt=len(output.outputs[0].token_ids)
            tt=task["task_type"]
            if tt=="draft":
                steps=split_steps(g);pred=boxed(g)
                rec={"task_type":"draft","dataset":task["dataset"],
                     "doc_id":task["doc_id"],"draft_idx":task["draft_idx"],
                     "mode":task["mode"],"question":task["question"],
                     "gold_answer":task["gold_answer"],
                     "draft_response":g,"draft_steps":steps,
                     "draft_n_steps":len(steps),"draft_answer":pred,
                     "draft_correct":float(norm(pred)==norm(task["gold_answer"])),
                     "draft_tokens":nt}
            elif tt=="suffix":
                full=task["prefix_text"]+("\n\n"+g if g else"")
                pred=boxed(full)
                rec={"task_type":"suffix","dataset":task["dataset"],
                     "doc_id":task["doc_id"],"alpha":task["alpha"],
                     "suffix_idx":task["suffix_idx"],"draft_idx":task["draft_idx"],
                     "gold_answer":task["gold_answer"],"pred_answer":pred,
                     "exact_match":float(norm(pred)==norm(task["gold_answer"])),
                     "suffix_tokens":nt,"rollback_step":task["rollback_step"],
                     "draft_n_steps":task["draft_n_steps"]}
            else:
                pred=boxed(g)
                rec={"task_type":"full_sc","dataset":task["dataset"],
                     "doc_id":task["doc_id"],"sample_idx":task["sample_idx"],
                     "gold_answer":task["gold_answer"],"pred_answer":pred,
                     "exact_match":float(norm(pred)==norm(task["gold_answer"])),
                     "n_tokens":nt}
            fout.write(json.dumps(rec,ensure_ascii=False)+"\n")
    print(f"[Shard {args._sid}] Done")


# ── coordinator ───────────────────────────────────────────────────────────

def launch(args,tasks,shard_dir,gpu_ids):
    if not tasks:return[]
    ns=len(gpu_ids)
    st=[[]for _ in range(ns)]
    for i,t in enumerate(tasks):st[i%ns].append(t)

    script=str(Path(__file__).resolve())
    procs,outs=[],[]
    for si,gid in enumerate(gpu_ids):
        if not st[si]:continue
        tf=shard_dir/f"t_{si}.json"
        tf.write_text(json.dumps(st[si],ensure_ascii=False),encoding="utf-8")
        so=shard_dir/f"s_{si}.jsonl"
        outs.append(so)
        cmd=[sys.executable,script,
             "--model-id",args.model_id,
             "--temperature",str(args.temperature),
             "--top-p",str(args.top_p),
             "--max-tokens",str(args.max_tokens),
             "--gpu-memory-utilization",str(args.gpu_memory_utilization),
             "--max-model-len",str(args.max_model_len),
             "--_sid",str(si),"--_out",str(so),
             "--_gpu",gid,"--_tf",str(tf)]
        env=os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]=gid
        env["TOKENIZERS_PARALLELISM"]="false"
        print(f"  Shard {si} GPU {gid} ({len(st[si])} tasks)")
        p=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        procs.append((si,gid,p))

    failed=[]
    for si,gid,p in procs:
        stdout,_=p.communicate()
        txt=(stdout.decode("utf-8",errors="replace")if stdout else"")
        rc=p.returncode
        lines=txt.strip().splitlines()
        tail="\n".join(lines[-5:])if lines else"(no output)"
        print(f"  Shard {si} (GPU {gid}) exit={rc}\n{tail}")
        if rc!=0:
            failed.append(si)
            if len(lines)>5:print("...\n"+"\n".join(lines[-20:]))

    if failed:
        print(f"ERROR: shards {failed} failed!");sys.exit(1)

    recs=[]
    for sf in outs:
        if sf.exists():
            recs.extend([json.loads(l)for l in sf.read_text("utf-8").splitlines()if l.strip()])
            sf.unlink(missing_ok=True)
    for si in range(ns):
        (shard_dir/f"t_{si}.json").unlink(missing_ok=True)
    return recs


# ── main ──────────────────────────────────────────────────────────────────

def main():
    args=parse_args()
    if args._sid>=0:
        run_shard(args);return

    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    sd=out/"_shards";sd.mkdir(parents=True,exist_ok=True)
    gpus=[g.strip()for g in args.gpus.split(",")if g.strip()]
    nd_list=[int(x)for x in args.n_drafts_list.split(",")]
    alphas=[float(x)for x in args.alphas.split(",")]
    Ks=[int(x)for x in args.Ks.split(",")]
    K_max=max(Ks)
    nd_max=max(nd_list)

    print("="*60)
    print("Draft-Alpha Sweep (GSM8K + MATH500)")
    print(f"  n_drafts: {nd_list}, alphas: {alphas}, Ks: {Ks}")
    print(f"  GPUs: {gpus}")
    print("="*60)

    t0=time.time()

    # Load data
    qs_g=load_gsm8k(args.n_gsm8k,args.seed)
    qs_m=load_math500(args.n_math500,args.seed)
    questions=qs_g+qs_m
    nq=len(questions)
    print(f"  {len(qs_g)} GSM8K + {len(qs_m)} MATH500 = {nq} questions")

    # Save question list for reproducibility
    (out/"questions.json").write_text(json.dumps(questions,ensure_ascii=False,indent=1),encoding="utf-8")

    # ── Phase 1: All drafts (nd_max: 1 greedy + nd_max-1 sampled) ────────
    draft_path=out/"drafts.jsonl"
    existing_drafts=load_jsonl(draft_path)
    done_dk=set(f"{d['doc_id']}_{d['draft_idx']}"for d in existing_drafts)

    draft_tasks=[]
    for q in questions:
        for di in range(nd_max):
            key=f"{q['doc_id']}_{di}"
            if key in done_dk:continue
            draft_tasks.append({
                "task_type":"draft","dataset":q["dataset"],
                "doc_id":q["doc_id"],"draft_idx":di,
                "mode":"greedy"if di==0 else"sampled",
                "question":q["question"],"gold_answer":q["gold_answer"],
                "prompt":prompt(q["question"]),
            })

    print(f"\n[Phase 1] Drafts: {len(existing_drafts)} cached, {len(draft_tasks)} pending (nd_max={nd_max})")
    if draft_tasks:
        t1=time.time()
        new=launch(args,draft_tasks,sd,gpus)
        append_jsonl(draft_path,new)
        existing_drafts.extend(new)
        print(f"  Done in {fmt(time.time()-t1)}")

    # Index drafts: doc_id -> {draft_idx -> record}
    drafts_idx=defaultdict(dict)
    for d in existing_drafts:
        drafts_idx[d["doc_id"]][d["draft_idx"]]=d

    greedy_correct=sum(1 for q in questions if drafts_idx[q["doc_id"]].get(0,{}).get("draft_correct",0))
    greedy_acc=greedy_correct/max(nq,1)
    greedy_tokens=sum(drafts_idx[q["doc_id"]].get(0,{}).get("draft_tokens",0)for q in questions)
    greedy_tpq=greedy_tokens/max(nq,1)
    print(f"  Greedy acc: {greedy_acc:.4f}, avg tokens: {greedy_tpq:.0f}")

    # ── Phase 2: All suffixes (for nd_max drafts, all alphas, K_max-nd_max suffixes per draft) ──
    # For a given (nd, K): n_suffix_total = K - nd, split across nd drafts
    # We generate for nd_max drafts and K_max, then subset at vote time.
    n_sfx_per_draft=math.ceil((K_max-1)/nd_max)  # generous upper bound

    sfx_path=out/"suffixes.jsonl"
    existing_sfx=load_jsonl(sfx_path)
    done_sk=set(f"{s['doc_id']}_{s['draft_idx']}_{s['alpha']}_{s['suffix_idx']}"for s in existing_sfx)

    sfx_tasks=[]
    skipped=0
    for q in questions:
        did=q["doc_id"]
        for di in range(nd_max):
            d=drafts_idx[did].get(di)
            if d is None:continue
            T=d["draft_n_steps"]
            if T<2:skipped+=1;continue
            for alpha in alphas:
                b=max(1,math.ceil(alpha*T))
                if b>=T:b=T-1
                if b<1:b=1
                prefix="\n\n".join(d["draft_steps"][:b])
                p=prompt(q["question"])+prefix+"\n\n"
                for si in range(n_sfx_per_draft):
                    key=f"{did}_{di}_{alpha}_{si}"
                    if key in done_sk:continue
                    sfx_tasks.append({
                        "task_type":"suffix","dataset":q["dataset"],
                        "doc_id":did,"alpha":alpha,"suffix_idx":si,
                        "draft_idx":di,"gold_answer":q["gold_answer"],
                        "prefix_text":prefix,"prompt":p,
                        "rollback_step":b,"draft_n_steps":T,
                        "mode":"sampled",
                    })

    print(f"\n[Phase 2] Suffixes: {len(existing_sfx)} cached, {len(sfx_tasks)} pending")
    if sfx_tasks:
        t1=time.time()
        new=launch(args,sfx_tasks,sd,gpus)
        append_jsonl(sfx_path,new)
        existing_sfx.extend(new)
        print(f"  Done in {fmt(time.time()-t1)}")

    # ── Phase 3: Full SC samples (K_max) ─────────────────────────────────
    sc_path=out/"full_sc.jsonl"
    existing_sc=load_jsonl(sc_path)
    done_sc=set(f"{r['doc_id']}_{r['sample_idx']}"for r in existing_sc)

    sc_tasks=[]
    for q in questions:
        p=prompt(q["question"])
        for si in range(K_max):
            key=f"{q['doc_id']}_{si}"
            if key in done_sc:continue
            sc_tasks.append({
                "task_type":"full_sc","dataset":q["dataset"],
                "doc_id":q["doc_id"],"sample_idx":si,
                "gold_answer":q["gold_answer"],"prompt":p,
                "mode":"sampled",
            })

    print(f"\n[Phase 3] Full SC: {len(existing_sc)} cached, {len(sc_tasks)} pending")
    if sc_tasks:
        t1=time.time()
        new=launch(args,sc_tasks,sd,gpus)
        append_jsonl(sc_path,new)
        existing_sc.extend(new)
        print(f"  Done in {fmt(time.time()-t1)}")

    # ── Phase 4: Vote + LTE score ────────────────────────────────────────
    print(f"\n[Phase 4] Computing votes and LTE scores")

    # Index suffixes
    sfx_idx=defaultdict(list)
    for s in existing_sfx:
        sfx_idx[f"{s['doc_id']}_{s['draft_idx']}_{s['alpha']}"].append(s)

    # Index SC
    sc_idx=defaultdict(list)
    for r in existing_sc:
        sc_idx[r["doc_id"]].append(r)

    results=[]

    for ds_name in ["gsm8k","math500","all"]:
        if ds_name=="all":
            qs_sub=questions
        else:
            qs_sub=[q for q in questions if q["dataset"]==ds_name]
        if not qs_sub:continue
        nqs=len(qs_sub)

        g_acc=sum(1 for q in qs_sub if drafts_idx[q["doc_id"]].get(0,{}).get("draft_correct",0))/nqs
        g_tpq=sum(drafts_idx[q["doc_id"]].get(0,{}).get("draft_tokens",0)for q in qs_sub)/nqs

        # Full SC for each K
        for K in Ks:
            nc,tt=0,0
            for q in qs_sub:
                samples=sc_idx.get(q["doc_id"],[])
                samples_k=[s for s in samples if s["sample_idx"]<K]
                answers=[s["pred_answer"]for s in samples_k]
                tt+=sum(s["n_tokens"]for s in samples_k)
                if norm(vote(answers))==norm(q["gold_answer"]):nc+=1
            acc=nc/nqs;tpq=tt/nqs
            cm=tpq/g_tpq if g_tpq>0 else 999
            lte=(acc-g_acc)/cm if cm>1 else 0
            results.append({"dataset":ds_name,"method":"FullSC","n_drafts":0,
                           "K":K,"alpha":"-","accuracy":acc,"tpq":tpq,
                           "gain":acc-g_acc,"multiplier":cm,"LTE":lte})

        # Late Rollback for each (nd, alpha, K)
        for nd in nd_list:
            for K in Ks:
                n_sfx_total=K-nd
                if n_sfx_total<1:continue
                sfx_per_d=[n_sfx_total//nd]*nd
                for i in range(n_sfx_total%nd):sfx_per_d[i]+=1

                for alpha in alphas:
                    nc,tt=0,0
                    for q in qs_sub:
                        did=q["doc_id"]
                        answers=[]
                        draft_tok=0
                        for di in range(nd):
                            d=drafts_idx[did].get(di)
                            if d:
                                answers.append(d["draft_answer"])
                                draft_tok+=d["draft_tokens"]
                        sfx_tok=0
                        for di in range(nd):
                            key=f"{did}_{di}_{alpha}"
                            ss=sfx_idx.get(key,[])
                            ss_k=[s for s in ss if s["suffix_idx"]<sfx_per_d[di]]
                            for s in ss_k:
                                answers.append(s["pred_answer"])
                                sfx_tok+=s["suffix_tokens"]
                        tt+=draft_tok+sfx_tok
                        if norm(vote(answers))==norm(q["gold_answer"]):nc+=1

                    acc=nc/nqs;tpq=tt/nqs
                    cm=tpq/g_tpq if g_tpq>0 else 999
                    lte=(acc-g_acc)/cm if cm>1 else 0
                    results.append({"dataset":ds_name,"method":"LateRollback",
                                   "n_drafts":nd,"K":K,"alpha":alpha,
                                   "accuracy":acc,"tpq":tpq,
                                   "gain":acc-g_acc,"multiplier":cm,"LTE":lte})

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{'Dataset':<8} {'Method':<14} {'nD':>3} {'K':>3} {'alpha':>6} "
          f"{'Acc':>7} {'Tok/Q':>7} {'Gain':>7} {'xCost':>6} {'LTE':>8}")
    print(f"{'-'*90}")

    for ds in ["gsm8k","math500","all"]:
        ds_res=sorted([r for r in results if r["dataset"]==ds],
                      key=lambda r:(-r["LTE"]))
        if not ds_res:continue
        for r in ds_res[:30]:
            print(f"{r['dataset']:<8} {r['method']:<14} {r['n_drafts']:>3} "
                  f"{r['K']:>3} {str(r['alpha']):>6} "
                  f"{r['accuracy']:>7.4f} {r['tpq']:>7.0f} "
                  f"{r['gain']:>+7.4f} {r['multiplier']:>6.1f} "
                  f"{r['LTE']:>8.5f}")
        print()

    # Save
    summary={"greedy_acc_gsm8k":sum(1 for q in qs_g if drafts_idx[q["doc_id"]].get(0,{}).get("draft_correct",0))/max(len(qs_g),1),
             "greedy_acc_math500":sum(1 for q in qs_m if drafts_idx[q["doc_id"]].get(0,{}).get("draft_correct",0))/max(len(qs_m),1),
             "n_gsm8k":len(qs_g),"n_math500":len(qs_m),
             "results":results}
    (out/"sweep_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    elapsed=time.time()-t0
    print(f"Total time: {fmt(elapsed)}")
    print(f"Summary -> {out/'sweep_summary.json'}")

    try:sd.rmdir()
    except:pass
    print("Done.")


if __name__=="__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    main()
