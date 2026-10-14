// Oracle TypeScript du test différentiel (tests/diff_vs_ts_validator.py).
//
// Charge le VRAI validateur du jeu (packages/shared/src/move-validator.ts) et, pour
// chaque position fournie :
//   - énumère tous les coups que le serveur accepterait (complétude du masque Python),
//   - rejoue validateHumanAction (game.ts) sur chaque message que le bot enverrait.
// Fonctions pures uniquement : aucun backend lancé, aucune écriture en base.
//
// Usage : npx tsx tests/ts_oracle.mts <mercury/packages/shared/src> <in.json> <out.json>

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [sharedDir, inFile, outFile] = process.argv.slice(2);
const load = (f: string) => import(pathToFileURL(path.join(sharedDir!, f)).href);

const { getLegalAction, getLegalSplit7Action, findLegalMoveForCard, findSolidaireEntry } =
    await load('move-validator.ts');
const { getTeammateColor } = await load('teams.ts');
const { MAIN_PATH, HOME_POSITIONS } = await load('board-config.ts');
const { ENTER_CARDS } = await load('constants.ts');

// Miroir de Game.validateHumanAction (backend/src/game/game.ts).
function validate(action: any, hand: any[], ctx: any): any {
    const solidaire = findSolidaireEntry(hand, ctx);
    if (action.type === 'discard') {
        if (solidaire) return null;
        if (hand.some(c => findLegalMoveForCard(c, ctx) !== null)) return null;
        return { type: 'discard' };
    }
    const card = action.cardPlayed?.[0];
    if (!card) return null;
    if (!hand.some(c => c.id === card.id)) return null;
    if (solidaire) {
        if (!ENTER_CARDS.includes(card.value)) return null;
        const teammate = ctx.teammateColor;
        const isTeammateReserve = HOME_POSITIONS[teammate].includes(action.from)
            && ctx.marblesByColor[teammate].includes(action.from);
        if (!isTeammateReserve) return null;
        return { ...solidaire, from: action.from, cardPlayed: [card] };
    }
    if (card.value === '7' && action.splitFrom !== undefined && action.splitFrom !== 0) {
        const fromIdx = MAIN_PATH.indexOf(action.from);
        const toIdx = MAIN_PATH.indexOf(action.to);
        if (fromIdx === -1 || toIdx === -1) return null;
        const steps1 = (toIdx - fromIdx + MAIN_PATH.length) % MAIN_PATH.length;
        return getLegalSplit7Action(card, action.from, steps1, action.splitFrom, ctx);
    }
    const target = card.value === 'J' ? action.to : undefined;
    return getLegalAction(card, action.from, ctx, target);
}

const cases = JSON.parse(fs.readFileSync(inFile!, 'utf8'));
const out = cases.map((c: any) => {
    // L'ordre des clés = ordre des joueurs côté serveur (colorAtPosition en dépend).
    const marblesByColor = Object.fromEntries(c.order.map((col: string) => [col, c.marbles[col]]));
    const all: number[] = (Object.values(marblesByColor) as number[][]).flat();
    const ctx = {
        ownMarbles: [...marblesByColor[c.controlled]],
        allMarbles: all,
        playerColor: c.controlled,
        marblesByColor,
        invincibleMarblesByColor: Object.fromEntries(c.order.map((col: string) => [col, c.inv[col]])),
        teammateColor: getTeammateColor(c.controlled),
    };

    const legal: string[] = [];
    c.hand.forEach((card: any, ci: number) => {
        if (card.value === 'J') {
            for (const from of all) for (const to of all) {
                const a = getLegalAction(card, from, ctx, to);
                if (a) legal.push(['swap', ci, from, to].join('|'));
            }
            return;
        }
        for (const from of all) {
            const a = getLegalAction(card, from, ctx);
            if (a) legal.push([a.type, ci, from, a.to].join('|'));
        }
        if (card.value !== '7') return;
        for (const from of all) for (let s = 1; s <= 6; s++) for (const from2 of all) {
            const a = getLegalSplit7Action(card, from, s, from2, ctx);
            if (a) legal.push(['split', ci, from, s, from2, a.type, a.to, a.splitType, a.splitTo].join('|'));
        }
    });

    const solidaire = findSolidaireEntry(c.hand, ctx);
    return {
        legal: legal.sort(),
        solidaire: solidaire ? [solidaire.cardPlayed[0].value, solidaire.from, solidaire.to] : null,
        wire: c.wire.map((w: any) => {
            const a = validate(w, c.hand, ctx);
            return a && { type: a.type, to: a.to ?? null, splitTo: a.splitTo ?? null, splitType: a.splitType ?? null };
        }),
    };
});
fs.writeFileSync(outFile!, JSON.stringify(out));
