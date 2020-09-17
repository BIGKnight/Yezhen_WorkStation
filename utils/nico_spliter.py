import os


# used to remove all .DS_Store files from mac os.
def file_fliter(root_path):
    for home, dirs, files in os.walk(root_path):
        for file_name in files:
            if file_name.startswith("."):
                print(os.path.join(home, file_name))
                try:
                    os.remove(os.path.join(home, file_name))
                except:
                    print('Wrong path...')


def read_dataset(root_path):
    ds = {}
    domain_freq = {}
    for category in os.listdir(root_path):
        ds[category] = {}
        # new a dict to store each category's domain information.
        in_cate = {}
        for domain in os.listdir(os.path.join(root_path, category)):
            if domain not in domain_freq:
                domain_freq[domain] = 0
            domain_freq[domain] += 1
            # print(domain)
            in_cate[domain] = []
            for file in os.listdir(os.path.join(root_path, category, domain)):
                in_cate[domain].append(os.path.join(category, domain, file))
                # print(in_cate[domain][-1])
            ds[category] = in_cate

    return ds


def domain_spliter(ds, source_domain, target_domain):
    source = []
    for category in ds:
        sub_area_images = ds[category][source_domain]
        source.extend(sub_area_images)

    pass
    import numpy as np
    prepared_target = {}
    for n_pct in [1, 3, 10]:
        def select_by_category(domain):
            ret_arrays = []
            for category in ds:
                if int(len(ds[category][domain]) * 0.01 * n_pct) >= 1:
                    selection = int(len(ds[category][domain]) * 0.01 * n_pct)
                else:
                    selection = 1
                selected_arrays = np.random.choice(ds[category][domain], selection)
                ret_arrays.extend(selected_arrays.tolist())
            return ret_arrays

        prepared_target[n_pct] = select_by_category(target_domain)

    return source, prepared_target


def write_to_txt(source: list, save_path: str, save_name: str):
    with open(os.path.join(save_path, save_name), 'w') as fp:
        for item in source:
            fp.writelines(item + '\n')


if __name__ == '__main__':
    # file_fliter("/home/v-boli4/codebases/external_datasets/NICO-Traffic")
    ds = read_dataset('/home/v-boli4/codebases/external_datasets/NICO-ANIMAL')
    ds.pop('bear', None)
    ds.pop('bird', None)


    def pipeline(source_domain, target_domain):
        source, prepared_target = domain_spliter(ds, source_domain, target_domain)
        write_to_txt(source, '/home/v-boli4/codebases/DA_Codebase/utils', 'source_{}.txt'.format(source_domain))
        for pct in prepared_target:
            write_to_txt(prepared_target[pct], '/home/v-boli4/codebases/DA_Codebase/utils',
                         '{}pct_{}.txt'.format(pct, target_domain))

    pipeline('on grass', 'on snow')
    pipeline('on snow', 'on grass')
