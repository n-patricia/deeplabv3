from dataloaders.datasets import pascal, coco
import torch.utils.data as data

def make_data_loader(args, **kwargs):
    if args.dataset_name == "pascal":
        train_set = pascal.VOCSegmentation(args, base_dir=args.dataset_dir, split="train")
        val_set = pascal.VOCSegmentation(args, base_dir=args.dataset_dir, split="val")

        num_class = train_set.NUM_CLASS
        train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
    elif args.dataset_name == "coco":
        train_set = coco.COCOSegmentation(args, base_dir=args.dataset_dir, split="train")
        val_set = coco.COCOSegmentation(args, base_dir=args.dataset_dir, split="val")

        num_class = train_set.NUM_CLASS
        train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return NotImplementedError
    else:
        return NotImplementedError

    return train_loader, val_loader, test_loader, num_class
