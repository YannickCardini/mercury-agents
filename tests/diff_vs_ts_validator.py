"""
Test différentiel : mercury_legal_moves.py contre le VRAI validateur du jeu
(mercury/packages/shared/src/move-validator.ts), sur des positions aléatoires.

Pour chaque position, vérifie dans les deux sens :
  - SÛRETÉ     : chaque coup du masque Python, une fois encodé par build_server_message,
                 est accepté par validateHumanAction ET produit l'effet que le
                 planificateur a simulé (type, case d'arrivée, 2e demi-coup du 7).
                 Un écart ici = actionRejected en partie = tour perdu au timeout.
  - COMPLÉTUDE : tout coup accepté par le serveur existe dans le masque Python (sinon
                 le bot croit devoir défausser alors que le serveur dit canDiscard=false).

Aucun backend lancé, aucune écriture en base : seules les fonctions pures du validateur
sont exécutées (via tests/ts_oracle.mts et le tsx du repo du jeu).

Usage : python tests/diff_vs_ts_validator.py [nb_positions] [seed]
        MERCURY_REPO=/chemin/vers/mercury pour un repo du jeu ailleurs que ~/Documents.
"""

import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mercury_legal_moves import (  # noqa: E402
    get_legal_mask, build_server_message, get_controlled_color, get_teammate_color,
    discard_idx_for, SOLIDAIRE_IDX,
    MAIN_PATH, HOME_POSITIONS, START_POSITIONS, ARRIVAL_POSITIONS, ALL_COLORS,
)

MERCURY_REPO = pathlib.Path(os.environ.get(
    "MERCURY_REPO", pathlib.Path.home() / "Documents" / "mercury"))

# Cartes surreprésentées : celles dont les règles sont les plus riches.
CARD_VALUES = (['7'] * 4 + ['J'] * 4 + ['K', 'A', 'Joker', '4'] * 2
               + ['2', '3', '5', '6', '8', '9', '10', 'Q'])


def random_case(rng: random.Random) -> dict:
    """Position aléatoire, volontairement dense autour des cases de départ (pieux
    invincibles, promotions, pions d'équipe à moins de 7 cases les uns des autres)."""
    hot = [MAIN_PATH[(MAIN_PATH.index(s) + d) % len(MAIN_PATH)]
           for s in START_POSITIONS.values() for d in range(-9, 8)]
    taken: set[int] = set()
    marbles, inv = {}, {}
    finished = rng.choice(ALL_COLORS) if rng.random() < 0.15 else None
    for color in ALL_COLORS:
        positions = []
        homes, arrivals = list(HOME_POSITIONS[color]), list(ARRIVAL_POSITIONS[color])
        for _ in range(4):
            r = rng.random()
            if color == finished or r < 0.2:
                # Zone d'arrivée remplie par le fond (comme en partie), parfois non.
                pos = arrivals.pop() if rng.random() < 0.8 else arrivals.pop(0)
            elif r < 0.4:
                pos = homes.pop(0)
            else:
                pool = hot if rng.random() < 0.7 else MAIN_PATH
                pos = next(p for p in iter(lambda: rng.choice(pool), None)
                           if p not in taken)
            taken.add(pos)
            positions.append(pos)
        marbles[color] = positions
        start = START_POSITIONS[color]
        inv[color] = [start] if start in positions and rng.random() < 0.6 else []

    order = list(ALL_COLORS)
    if rng.random() < 0.3:      # l'ordre des joueurs ne doit rien changer
        rng.shuffle(order)
    player = finished if finished and rng.random() < 0.7 else rng.choice(ALL_COLORS)
    hand = [{'id': f'c{i}', 'value': v, 'suit': '♠'}
            for i, v in enumerate(rng.choices(CARD_VALUES, k=rng.randint(1, 5)))]
    return {'order': order, 'marbles': marbles, 'inv': inv, 'hand': hand,
            'controlled': get_controlled_color(player, marbles[player])}


def python_view(case: dict) -> dict:
    """Masque Python d'une position : coups (même encodage que l'oracle), messages
    wire et effets attendus."""
    color = case['controlled']
    teammate = get_teammate_color(color)
    mbc = {c: case['marbles'][c] for c in case['order']}   # ordre serveur
    hand = case['hand']
    mask, actions = get_legal_mask(hand, mbc[color], color, mbc,
                                   invincible_by_color=case['inv'],
                                   can_discard=False, teammate_color=teammate)
    legal, wire, expected, solidaire = [], [], [], None
    for idx, ok in enumerate(mask):
        if not ok:
            continue
        msg = build_server_message(idx, hand, mbc[color], color, mbc,
                                   invincible_by_color=case['inv'],
                                   teammate_color=teammate,
                                   legal=(mask, actions))['action']
        wire.append(msg)
        if idx == discard_idx_for(teammate):
            expected.append({'type': 'discard', 'to': None,
                             'splitTo': None, 'splitType': None})
            continue
        card, a = actions[idx]
        expected.append({'type': a['type'], 'to': a['to'],
                         'splitTo': a.get('splitTo'), 'splitType': a.get('splitType')})
        ci = next(i for i, c in enumerate(hand) if c is card)
        if idx == SOLIDAIRE_IDX:
            solidaire = [card['value'], a['from'], a['to']]
        elif a['type'] == 'swap':
            legal.append(f"swap|{ci}|{a['from']}|{a['to']}")
        elif 'splitFrom' in a:
            legal.append(f"split|{ci}|{a['from']}|{a['steps1']}|{a['splitFrom']}|"
                         f"{a['type']}|{a['to']}|{a['splitType']}|{a['splitTo']}")
        else:
            legal.append(f"{a['type']}|{ci}|{a['from']}|{a['to']}")
    return {'legal': sorted(legal), 'wire': wire, 'expected': expected,
            'solidaire': solidaire}


def run_oracle(cases: list[dict]) -> list[dict]:
    shared = MERCURY_REPO / "packages" / "shared" / "src"
    backend = MERCURY_REPO / "backend"
    if not shared.is_dir():
        sys.exit(f"repo du jeu introuvable ({shared}) — définir MERCURY_REPO")
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = pathlib.Path(tmp, "in.json"), pathlib.Path(tmp, "out.json")
        fin.write_text(json.dumps(cases))
        subprocess.run(["npx", "tsx", str(HERE / "ts_oracle.mts"),
                        str(shared), str(fin), str(fout)],
                       cwd=backend, check=True)
        return json.loads(fout.read_text())


def describe(case: dict, a: str) -> str:
    """Catégorie lisible d'un coup en écart (pour le résumé)."""
    kind, ci = a.split('|')[:2]
    return f"{case['hand'][int(ci)]['value']}:{kind}"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = random.Random(seed)
    cases = [random_case(rng) for _ in range(n)]
    views = [python_view(c) for c in cases]
    oracle = run_oracle([{**c, 'wire': v['wire']} for c, v in zip(cases, views)])

    bad_cases, summary, shown = 0, {}, 0

    def note(label: str, case: dict, detail: str) -> None:
        nonlocal shown
        summary[label] = summary.get(label, 0) + 1
        if shown < 8:
            shown += 1
            print(f"- {label}: {detail}\n    {json.dumps(case, ensure_ascii=False)}")

    for case, py, ts in zip(cases, views, oracle):
        problems = 0
        for w, exp, got in zip(py['wire'], py['expected'], ts['wire']):
            if got is None:
                problems += 1
                note(f"REJETÉ par le serveur ({w['cardPlayed'][0]['value']}:{w['type']})",
                     case, json.dumps({k: w.get(k) for k in
                                       ('type', 'from', 'to', 'splitFrom')}))
            elif got != exp:
                problems += 1
                note(f"effet différent ({w['cardPlayed'][0]['value']}:{w['type']})",
                     case, f"python={exp} serveur={got}")
        for a in sorted(set(ts['legal']) - set(py['legal'])):
            problems += 1
            note(f"coup serveur ABSENT du masque ({describe(case, a)})", case, a)
        if py['solidaire'] != ts['solidaire']:
            problems += 1
            note("entrée solidaire différente", case,
                 f"python={py['solidaire']} serveur={ts['solidaire']}")
        bad_cases += bool(problems)

    n_moves = sum(len(v['wire']) for v in views)
    print(f"\n{n} positions, {n_moves} coups comparés (seed={seed})")
    for label, count in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"  {count:6d}  {label}")
    print(f"=> {n - bad_cases}/{n} positions identiques au serveur")
    sys.exit(0 if bad_cases == 0 else 1)


if __name__ == "__main__":
    main()
