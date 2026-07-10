"""
Génère la fixture de NON-RÉGRESSION 1v3 : des états de référence variés et le masque
légal calculé par mercury_legal_moves.get_legal_mask AU MOMENT DE LA GÉNÉRATION.

⚠️ NE PAS RÉGÉNÉRER : la fixture committée a été produite avec la version PRÉ-2v2 du
module — c'est ça qui fait sa valeur de référence. test_regression_1v3.py rejoue les
mêmes entrées avec teammate_color=None et exige un masque identique, à une divergence
documentée près (correctif du split-7 promote-first, cf. docstring du test). Régénérer
avec le code courant absorberait silencieusement toute régression future.
"""

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mercury_legal_moves import (  # noqa: E402
    get_legal_mask, MAIN_PATH, HOME_POSITIONS, ARRIVAL_POSITIONS,
    START_POSITIONS, ALL_COLORS,
)

CARD_VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'Joker']

OUT = pathlib.Path(__file__).parent / "fixture_1v3_masks.json"


def random_state(rng: random.Random) -> dict:
    """État aléatoire cohérent : positions uniques, invincibles seulement sur leur start."""
    taken: set[int] = set()
    marbles_by_color: dict[str, list[int]] = {}
    invincible_by_color: dict[str, list[int]] = {}
    for color in ALL_COLORS:
        positions: list[int] = []
        inv: list[int] = []
        homes = list(HOME_POSITIONS[color])
        rng.shuffle(homes)
        for _ in range(4):
            zone = rng.random()
            if zone < 0.35:                       # home
                pos = homes.pop()
            elif zone < 0.85:                     # main path
                pos = rng.choice([p for p in MAIN_PATH if p not in taken])
                if pos == START_POSITIONS[color] and rng.random() < 0.5:
                    inv.append(pos)
            else:                                 # arrivée
                free = [p for p in ARRIVAL_POSITIONS[color]
                        if p not in taken and p not in positions]
                pos = free[0] if free else homes.pop()
            if pos in taken:
                pos = homes.pop()
            taken.add(pos)
            positions.append(pos)
        marbles_by_color[color] = positions
        invincible_by_color[color] = inv

    hand = [{'id': f'c{i}', 'value': rng.choice(CARD_VALUES), 'suit': '♠'}
            for i in range(rng.randint(1, 5))]
    return {
        'hand': hand,
        'color': rng.choice(ALL_COLORS),
        'marbles_by_color': marbles_by_color,
        'invincible_by_color': invincible_by_color,
        'can_discard': rng.random() < 0.1,
    }


def main():
    rng = random.Random(20260709)
    cases = []
    for _ in range(300):
        st = random_state(rng)
        mask, _ = get_legal_mask(
            st['hand'], st['marbles_by_color'][st['color']], st['color'],
            st['marbles_by_color'],
            invincible_by_color=st['invincible_by_color'],
            can_discard=st['can_discard'],
        )
        cases.append({**st, 'mask_true_idx': [i for i, ok in enumerate(mask) if ok]})
    OUT.write_text(json.dumps(cases))
    n_legal = sum(len(c['mask_true_idx']) for c in cases)
    print(f"{len(cases)} états écrits dans {OUT} ({n_legal} actions légales au total)")


if __name__ == '__main__':
    main()
