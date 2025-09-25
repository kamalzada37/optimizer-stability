# train.py
import argparse, os, json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop, Adagrad
from data import get_mnist, inject_label_noise
from model import SmallCNN
from utils import set_seed, save_json

def get_optimizer(name, params, lr):
    if name == 'sgd': return SGD(params, lr=lr, momentum=0.9)
    if name == 'adam': return Adam(params, lr=lr)
    if name == 'rmsprop': return RMSprop(params, lr=lr)
    if name == 'adagrad': return Adagrad(params, lr=lr)
    raise ValueError(name)

def train_one_epoch(model, loader, opt, device, loss_fn, dtype):
    model.train()
    running_loss = 0.0
    for xb,yb in loader:
        xb = xb.to(device=device, dtype=dtype)
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

def eval_model(model, loader, device, dtype):
    model.eval()
    correct = 0
    tot = 0
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(device=device, dtype=dtype)
            yb = yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            tot += xb.size(0)
    return correct / tot

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64 if args.precision == 'float64' else torch.float32

    # Data
    train_ds = get_mnist('../data', train=True, download=True)
    train_ds = inject_label_noise(train_ds, args.noise, num_classes=10, seed=args.seed)
    test_ds = get_mnist('../data', train=False, download=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=False)

    # Model
    model = SmallCNN()
    model = model.to(device=device, dtype=dtype)
    loss_fn = nn.CrossEntropyLoss()

    opt = get_optimizer(args.optimizer, model.parameters(), args.lr)
    history = {'train_loss':[], 'test_acc':[]}
    meta = {'optimizer':args.optimizer, 'lr':args.lr, 'noise':args.noise, 'precision':args.precision, 'seed':args.seed, 'batch_size':args.batch_size, 'epochs':args.epochs}
    start = time.time()

    diverged = False
    for ep in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, opt, device, loss_fn, dtype)
        te_acc = eval_model(model, test_loader, device, dtype)
        history['train_loss'].append(tr_loss)
        history['test_acc'].append(te_acc)
        print(f"[{ep+1}/{args.epochs}] loss={tr_loss:.4f} test_acc={te_acc:.4f}")
        if not (tr_loss == tr_loss and tr_loss < 1e9):  # detect NaN or explosion
            diverged = True
            print("Diverged or NaN detected. Stopping.")
            break

    elapsed = time.time() - start
    out = {'meta': meta, 'history': history, 'diverged': diverged, 'elapsed_sec': elapsed}
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"res_{args.optimizer}_noise{int(100*args.noise)}_{args.precision}_seed{args.seed}.json")
    save_json(fname, out)

    # Optionally save checkpoint
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(out_dir, f"model_{args.optimizer}_noise{int(100*args.noise)}_{args.precision}_seed{args.seed}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd','adam','rmsprop','adagrad'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--precision', type=str, default='float32', choices=['float32','float64'])
    parser.add_argument('--outdir', type=str, default='../results')
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()
    main(args)
