import json, os
from pathlib import Path

ORIGINAL = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal")
OUT_ROOT = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3")
MAX_EPS  = 50

# Mapping task_index → istruzione originale (da tasks.jsonl)
ORIG_TASKS = {
    0: "put the bowl on the plate",
    1: "put the wine bottle on the rack",
    2: "open the top drawer and put the bowl inside",
    3: "put the cream cheese in the bowl",
    4: "put the wine bottle on top of the cabinet",
    5: "push the plate to the front of the stove",
    6: "turn on the stove",
    7: "put the bowl on the stove",
    8: "put the bowl on top of the cabinet",
    9: "open the middle drawer of the cabinet",
}

# task_index → lista di varianti L3 di training
L3_VARIATIONS = {
    9: [  # Task 1: Open middle drawer
        "Open the layer of the drawer below the top drawer",
        "Open the layer of the drawer above the bottom drawer",
    ],
    7: [  # Task 2: Put bowl on stove
        "Put the object in front of the wine bottle on the stove",
        "Put the object behind the plate on the stove",
        "Put the object to the right of cream cheese on the stove",
    ],
    4: [  # Task 3: Put wine bottle on top of cabinet (solo V3-V5)
        "Put the object in front of the rack on the top of the cabinet",
        "Put the object to the left of the rack on the top of the cabinet",
        "Put the object between the bowl and the rack on the top of the cabinet",
    ],
    2: [  # Task 4: Open top drawer and put bowl inside
        "Open the top layer of the drawer and put the object to the right of the cream cheese inside",
        "Open the top layer of the drawer and put the object in front of the wine bottle inside",
        "Open the top layer of the drawer and put the object behind the plate inside",
    ],
    8: [  # Task 5: Put bowl on top of cabinet
        "Put the object in front of the wine bottle on the top of the cabinet",
        "Put the object behind the plate on the top of the cabinet",
        "Put the object to the right of the cream cheese on the top of the cabinet",
    ],
    5: [  # Task 6: Push plate to front of stove
        "Push the object in front of the bowl to the front of the stove",
        "Push the object in front of the cabinet to the front of the stove",
        "Push the object left of the cabinet to the front of the stove",
    ],
    3: [  # Task 7: Put cream cheese in bowl
        "Put the object to the left of the plate on the bowl",
        "Put the object next to the stove on the bowl",
    ],
    6: [  # Task 8: Turn on stove
        "Turn on the object to the left of the wine bottle",
        "Turn on the object farthest from the cabinet",
    ],
    0: [  # Task 9: Put bowl on plate
        "Put the object in front of the wine bottle on the plate",
        "Put the object behind the plate on the plate",
        "Put the object to the right of the cream cheese on the plate",
    ],
    1: [  # Task 10: Put wine bottle on rack (solo V3-V5)
        "Put the object in front of the rack on the rack",
        "Put the object to the left of the rack on the rack",
        "Put the object between the bowl and the rack on the rack",
    ],
}

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def create_virtual_dataset(task_idx, var_idx, instruction, max_eps):
    name = f"task{task_idx:02d}_v{var_idx:02d}"
    out  = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    # Symlink ai dati originali (non duplicano nulla su disco)
    for d in ["data", "videos"]:
        link = out / d
        if not link.exists():
            os.symlink(ORIGINAL / d, link)

    # Filtra episodi per task_index e limita a max_eps
    all_eps  = load_jsonl(ORIGINAL / "meta" / "episodes.jsonl")
    filtered = [
        e for e in all_eps
        if e["tasks"][0].lower() == ORIG_TASKS[task_idx].lower()
    ][:max_eps]
    sel_idx  = {e["episode_index"] for e in filtered}

    meta = out / "meta"
    meta.mkdir(exist_ok=True)

    # tasks.jsonl: istruzione L3 sovrascritta
    with open(meta / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": instruction}) + "\n")

    # episodes.jsonl: solo gli episodi del task filtrato con istruzione L3
    total_frames = 0
    with open(meta / "episodes.jsonl", "w") as f:
        for e in filtered:
            f.write(json.dumps({
                "episode_index": e["episode_index"],
                "tasks": [instruction],
                "length": e["length"]
            }) + "\n")
            total_frames += e["length"]

    # info.json: aggiornato con i conteggi reali
    with open(ORIGINAL / "meta" / "info.json") as f:
        info = json.load(f)
    info.update({
        "total_episodes": len(filtered),
        "total_frames":   total_frames,
        "total_tasks":    1,
        "total_videos":   len(filtered) * 2,
        "splits":         {"train": f"0:{len(filtered)}"},
    })
    with open(meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # episodes_stats.jsonl: solo gli episodi selezionati
    stats_src = ORIGINAL / "meta" / "episodes_stats.jsonl"
    if stats_src.exists():
        with open(stats_src) as fi, open(meta / "episodes_stats.jsonl", "w") as fo:
            for line in fi:
                entry = json.loads(line)
                if entry.get("episode_index") in sel_idx:
                    fo.write(line)

    print(f"[OK] {name} | {len(filtered)} eps | '{instruction[:70]}'")
    return name

if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    created = []
    for task_idx in sorted(L3_VARIATIONS):
        for var_idx, instr in enumerate(L3_VARIATIONS[task_idx], start=1):
            created.append(create_virtual_dataset(task_idx, var_idx, instr, MAX_EPS))

    print(f"\nTotale virtual datasets creati: {len(created)}")
    print("Percorso output:", OUT_ROOT)