import pdb
from collections import defaultdict

clip16 = [[17, 22, 7],
          [33, 36, 8],
          [0, 1, 4, 11, 22, 50, ],
          [6, 38, 44],
          [4, 11, 10, 22, 33, 59, ],
          [12, 17, 50, ],
          [12, 31, 43, 42, 40, 50, ],
          [7, 6, 5, 16, 37, 41, 45, 55, 54, ],
          [3, 2, 0, 1, 5, 6, 19, 22, 24, 34, 33, 32, 36, 39, 50, ],
          [0, 1, 2, 11,  19, 21, 24, 25, 31, 33, 40, 47, 44, 48, 52, 56, ],
          [20, 33, 51, 55, 53, ],
          [2, 0, 4, 10, 9, 13, 25, 32, 41, ]]

clip16_vis = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1],
    [1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 1, 1]
]

vit16 = [[29, 30, 22],
         [31, 29],
         [5, 3, 36],
         [35, 27, 37],
         [5, 9, 12],
         [5, 10, 24],
         [1, 2, 6, 9, 20, 27],
         [1, 5, 4, 8, 14, 16, 17, 24, 37],
         [0, 3, 5, 19, 22, 28],
         [0, 25],
         [0, 9, 22, 20, 27, 37],
         [0, 16, 33]]

vit32 = [
    [], [], [], [], [],
    [32, 35, 25, ],
    [58, 45, 46, 42, 40, 32, 34, 17, 9, ],
    [55, 49, 44, 39, 25, 27, 17, 10, 0, ],
    [58, 52, 43, 27, 23, 18, 14, 15, 10, 9, 5, 2],
    [52, 48, 50, 47,  44, 40, 33, 34, 35, 29, 28, 24, 25, 16],
    [56, 51, 48, 45, 43, 42, 41, 40, 36, 35, 34, 31, 26, 22, 23, 18, 14, 13, 9, 10, 11, 6, 5,  ],
    []
]

vit32_tv = [
    [], [], [], [], [],
    [0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    []
]

clip32 = [
    [], [], [], [], [],
    [58, 47, 39, 32, 31, 9, 11, 3, ],
    [50, 51, 45, 33, 32, 31, ],
    [57, 53, 40, 42, 39, 38, ],
    [54, 47, 34, 16, 12, 13, ],
    [59, 47, 41, 27, 15, 14, ],
    [43, 39, 24, ],
    []
]

clip32_tv = [
    [], [], [], [], [],
    [0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [1, 0, 1]
]


def gen(arr):
    for l, layer in enumerate(arr):
        for f in layer:
            yield (l, f)


def generate(indices, per_row: int = 7, per_page: int = 4, tvs: list = None, high: str = '30.0', low: str = '1.0',
             folder: str = 'clip16'):
    with open(f'{folder}.row', 'w') as file:
        arr = [[]]
        for i, (l, f) in enumerate(gen(indices)):
            arr[-1].append([l, f])
            if i % per_row == per_row - 1:
                arr.append([])

        ctvs = defaultdict(list)
        for i, (_, tv) in enumerate(gen(tvs)):
            ctvs[(i // per_row)].append(tv)

        for i, sub in enumerate(arr):
            if i % per_page == 0:
                print('\\begin{figure*}[t]\centering\setlength\\tabcolsep{0pt}\\begin{tabularx}{\linewidth}{Y',
                      ('Y' * 2 * per_row), '}', sep='', file=file)
            for (l, f) in sub:
                print('& \salayer{', l, '}{', f, '}', end=' ', sep='', file=file)
            print('\\\\', file=file)

            print('\\', 'raisebox{1.0\\', 'totalheight}{ \\', 'rotatebox[origin=lB]{90}{Ours} }', sep='', end='', file=file)
            for j, (l, f) in enumerate(sub):
                print('& \savis{', l, '}{', f, '}{', high if ctvs[i][j] else low, '}{', folder, '}', end=' ', sep='', file=file)
            print('\\\\[-3.3pt]', file=file)

            print('\\', 'raisebox{0.7\\', 'totalheight}{ \\', 'rotatebox[origin=lB]{90}{Val} }', sep='', end='', file=file)
            for l, f in sub:
                print('& \saeval{', l, '}{', f, '}{', folder, '}', end=' ', sep='', file=file)
            print('\\\\[-3.3pt]', file=file)

            print('\\', 'raisebox{0.3\\', 'totalheight}{ \\', 'rotatebox[origin=lB]{90}{Train} }', sep='', end='', file=file)
            for l, f in sub:
                print('& \satrain{', l, '}{', f, '}{', folder, '}', end=' ', sep='', file=file)
            print('\\\\', file=file)

            # for l, f in sub:
            #     print('& \sacontent{}', end=' ', sep='')
            # print('\\\\')
            if i % per_page == per_page - 1:
                print('\end{tabularx} \caption{} \label{fig:zoom} \end{figure*}', file=file)
        print('\end{tabularx} \caption{} \label{fig:zoom} \end{figure*}', file=file)
        print('', file=file)


def generate2(indices, per_row: int = 7, per_page: int = 6, folder: str = 'vit16'):
    with open(f'{folder}.row', 'w') as file:
        arr = [[]]
        for i, (l, f) in enumerate(gen(indices)):
            arr[-1].append([l, f])
            if i % per_row == per_row - 1:
                arr.append([])

        for i, sub in enumerate(arr):
            if i % per_page == 0:
                print('\\begin{figure*}[t]\centering\setlength\\tabcolsep{0pt}\\begin{tabularx}{\linewidth}{Y',
                      ('Y' * 2 * per_row), '}', sep='', file=file)
            for (l, f) in sub:
                print('& \salayer{', l, '}{', f, '}', end=' ', sep='', file=file)
            print('\\\\', file=file)

            print('\\', 'raisebox{1.0\\', 'totalheight}{ \\', 'rotatebox[origin=lB]{90}{Ours} }', sep='', end='', file=file)
            for j, (l, f) in enumerate(sub):
                print('& \sbvis{', l, '}{', f, '}{', folder, '}', end=' ', sep='', file=file)
            print('\\\\[-3.3pt]', file=file)

            print('\\', 'raisebox{0.7\\', 'totalheight}{ \\', 'rotatebox[origin=lB]{90}{Val} }', sep='', end='', file=file)
            for l, f in sub:
                print('& \saeval{', l, '}{', f, '}{', folder, '}', end=' ', sep='', file=file)
            print('\\\\[-3.3pt]', file=file)

            if i % per_page == per_page - 1:
                print('\end{tabularx} \caption{} \label{fig:zoom} \end{figure*}', file=file)
        print('\end{tabularx} \caption{} \label{fig:zoom} \end{figure*}', file=file)
        print('', file=file)


def print35():
    print('mkdir -- vit35')
    print('mkdir -- vit35/vis')
    print('mkdir -- vit35/train')
    print('mkdir -- vit35/train_mask')
    print('mkdir -- vit35/eval')
    print('mkdir -- vit35/eval_mask')
    for l, f in gen(vit32):
        for tv in ['0.1', '1.0']:
            print(f'cp VisL{l}_F{f}_N35_TV{tv}/png_ImageSaver/0_final.png vit35/vis/{tv}_{l}_{f}.png')
            print(f'cp train_35/{l}_{f}.png vit35/train/{l}_{f}.png')
            print(f'cp eval_35/{l}_{f}.png vit35/eval/{l}_{f}.png')
            print(f'cp train_35_mask/{l}_{f}.png vit35/train_mask/{l}_{f}.png')
            print(f'cp eval_35_mask/{l}_{f}.png vit35/eval_mask/{l}_{f}.png')


def print98():
    print('mkdir -- clip32')
    print('mkdir -- clip32/vis')
    print('mkdir -- clip32/train')
    print('mkdir -- clip32/train_mask')
    print('mkdir -- clip32/eval')
    print('mkdir -- clip32/eval_mask')
    for l, f in gen(clip32):
        for tv in ['0.1', '1.0']:
            print(f'cp Clip_TV{tv}_L{l}_F{f}_N98/png_ImageSaver/0_final.png clip32/vis/{tv}_{l}_{f}.png')
            print(f'cp train_98/{l}_{f}.png clip32/train/{l}_{f}.png')
            print(f'cp eval_98/{l}_{f}.png clip32/eval/{l}_{f}.png')
            print(f'cp train_98_mask/{l}_{f}.png clip32/train_mask/{l}_{f}.png')
            print(f'cp eval_98_mask/{l}_{f}.png clip32/eval_mask/{l}_{f}.png')


def fake_arr(arr):
    return [[0 for _ in row] for row in arr]


if __name__ == '__main__':
    generate(clip16, tvs=clip16_vis)
    generate(vit32, tvs=vit32_tv, low='1.0', high='0.1', folder='vit32')
    # print35()
    # print98()
    generate(clip32, tvs=clip32_tv, low='0.1', high='1.0', folder='clip32')
    generate2(vit16)
    pass
