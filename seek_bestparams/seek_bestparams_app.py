from seek_bestparams_train import Train

def main():
    ls_models = ['densenet161', 'vgg16_bn']
    ls_n_hidden = [256, 512, 256*3, 1024, 512*3, 2048, 1024*3, 4096]

    ls_lr = [0.002, 0.004, 0.006]
    ls_n_epochs = [3]

    f = open('results.csv', 'w')
    f.write(f"{'index':<6}, {'model':<12}, {'n_hidden':<8}, {'lr':<6}, {'n_epochs':<10}, {'acc_train':<10}, {'acc_valid':<10}, {'time':<6}")
    f.close()

    train_index = 0
    for md in ls_models:
        for nh in ls_n_hidden:
            for lr in ls_lr:
                for ne in ls_n_epochs:
                    train_index += 1
                    print(f"\nidx: {train_index}, md:{md}, nh:{nh}, lr:{lr}, ne: {ne}")
                    train_obj = Train()
                    acc_train, acc_valid, elapsed_time = train_obj.main(md, nh, lr, ne)
                    record = f"\n{train_index:<6d}, {md:<12}, {nh:<8d}, {lr:<6.3f}, {ne:<10d}, {round(acc_train, 3):<10.3f}, {round(acc_valid, 3):<10.3f}, {round(elapsed_time/60, 3):<6.3f}"
                    f = open('results.csv', 'a')
                    f.write(record)
                    f.close()

if __name__ == '__main__':
    main()