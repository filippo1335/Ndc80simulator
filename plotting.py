# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import timeit
# import multiprocessing as mp

plt.rcParams['figure.dpi'] = 1000


"""def extractor_for_split(batch, n, chunksize):
    #for i in range(len(batch)):
    #batch_p["x"] = batch.loc[i + 2::n]
    d = {}
    id = batch.index[0] % chunksiz
    batch.reset_index(inplace=True, drop=True)
    for i in range(n):
        name = f'atom{i}'
        d[name] = batch.loc[i::n + 2]
        d[name].reset_index(inplace=True, drop=True)
        d[name].columns = [name]
        #print(d[name])
    df = pd.concat(d, axis=1)
    #df = pd.DataFrame(d.values(), index=d.keys())
    #df = pd.DataFrame.from_dict(d)
    print(df)
    df.to_csv(f"extraction/extracted{id}.csv", index=True)

    return df


def splitter(path, n):
    with open(path) as file:
        chunksiz = 5500
        reader = pd.read_table(file, chunksize=chunksiz, sep=' ', usecols=[1], skiprows=2, header=0, names=["hihi"])
        pool = mp.Pool(8)
        funclist = []
        frames = []
        #print(reader)
        for elem in reader:
            f = pool.apply_async(extractor_for_split(elem, n, chunksiz))
            funclist.append(f)
            #print(f.get(timeout=10))
        print(frames)
        for f in funclist:
            frames.append(f.get)
        #maindf = pd.concat(frames, axis = 1)
    #print(maindf)


if __name__ == '__main__':
    splitter("particle_coord_copy.txt", 53)
"""



def extractor(path, n):
    with open(path) as file:
        lines = file.readlines()
        d = {}
        #df = pd.DataFrame(columns = [f'atom{i}' for i in range(n)])
        for i in range(n):
            name = f'atom{i}'
            d[name] = []
            for line in lines[i + 2::n + 2]:
                x = line.split()
                d[name].append(float(x[1]))
            print(f"particle {i} finished")
    print("file closed, creating df")
    df = pd.DataFrame.from_dict(d, orient='columns')
    df["msd"] =+ ((df - df.loc[0]) ** 2).mean(axis=1)
    df.to_csv("extracted.csv", index=True)
    return df

def plotter(path, n, rerun, fign, seed, steps, dt, runName):
    print("plotter running")
    if rerun:
        coord = extractor(path, n)
    else:
        coord = pd.read_csv("extracted.csv")
    print("data loaded")
    time_ser = (coord.index).astype(float)*dt/1000
    fig, ax1= plt.subplots()
    ax1.plot(time_ser, coord['msd'], color='orange', linewidth=1, label='total msd')
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('MSD (Å^2)')
    ax1.title.set_text(f'{steps} steps, timestep=1fs, damp=1, seed={seed}')
    ax1.legend()
    ax1.grid(True)
    fig.show()
    fig.savefig(f"figures/{runName}/figure{fign}_{seed}.png")


# print(timeit.timeit("extractor('/Users/filiproch/lammps/dump/particle_coord_copy.txt', 53)", number=1, globals=globals()))

# plotter('/Users/filiproch/lammps/dump/particle_coord.txt', 10, False)



