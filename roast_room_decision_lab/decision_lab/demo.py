"""One visible synthetic decision: inspect scoring and a ruin cap."""
from .core import Case, SeatCard, aggregate, OPTIONS


def main():
    cards = (
        SeatCard('product', (8, 6, 4), 3, (False, False, False)),
        SeatCard('pricing', (9, 6, 5), 3, (False, False, False)),
        SeatCard('capital', (6, 7, 8), 2, (True, False, False)),
        SeatCard('operations', (8, 7, 6), 1, (False, False, False)),
    )
    before = aggregate(cards, ruin_cap=False, calls=4)
    after = aggregate(cards, ruin_cap=True, calls=4)
    print('Scripted option A/B/C, synthetic cards (not real advisors)')
    print('Relevance-weighted scores:')
    for i, option in enumerate(OPTIONS):
        val = sum(c.weight*c.scores[i] for c in cards)/sum(c.weight for c in cards)
        print(f'  {option}: {val:.2f}')
    print(f'No cap: {before.recommended} -> {before.label}')
    print(f'With ruin cap: {after.recommended} -> {after.label}')
    print('Notice: the recommended option is unchanged. The verdict label is capped.')

if __name__ == '__main__':
    main()
