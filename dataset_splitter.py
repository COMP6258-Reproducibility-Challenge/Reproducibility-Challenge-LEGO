import argparse
import os
import shutil


def filter_selected_classes(source_dir, target_dir, class_indices):

    os.makedirs(target_dir, exist_ok=True)

    class_folders = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

    class_indices = list(class_indices)


    selected_classes = [class_folders[i] for i in class_indices if i < len(class_folders)]

    for class_name in selected_classes:
        src_path = os.path.join(source_dir, class_name)
        dst_path = os.path.join(target_dir, class_name)
        shutil.copytree(src_path, dst_path)
        print(f"Copied: {class_name}")


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_dir', default='', type=str, help='source directory')
    parser.add_argument('--target_dir', default='', type=str, help='target directory')
    parser.add_argument('--start_ind', default='', type=int, help='start index')
    parser.add_argument('--end_ind', default='', type=int, help='end index')
    args = parser.parse_args()

    print('source_dir:', args.source_dir)

    filter_selected_classes(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        class_indices=range(args.start_ind, args.end_ind + 1)
    )

if __name__ == '__main__':
    main()


