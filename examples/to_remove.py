from multiprocessing import Pool


print(len(dataset))
print('generate random walks')
with Pool() as pool:
    for x1, x2, x3 in tqdm(pool.imap(generate_random_walks, dataset), total=len(dataset)):
        f_words.append(x1)
        f_graph_id.append(x2)
        labels.append(x3)
