import os, argparse, numpy as np
def gelu(x): return 0.5*x*(1.0 + np.tanh(0.7978845608*(x + 0.044715*(x**3))))
def write_bin(p,a,d): a.astype(d).tofile(p)
def write_i32(p,a): a.astype(np.int32).tofile(p)
ap=argparse.ArgumentParser()
ap.add_argument("--name",required=True); ap.add_argument("--outdir",default="cases")
ap.add_argument("--N",type=int,required=True); ap.add_argument("--B",type=int,required=True)
ap.add_argument("--d",type=int,required=True); ap.add_argument("--h",type=int,required=True)
ap.add_argument("--E",type=int,required=True); ap.add_argument("--seed",type=int,default=1)
args=ap.parse_args()
assert args.E % args.N == 0
E_local=args.E//args.N
rs=np.random.RandomState(args.seed)
Wg=0.2*(rs.rand(args.E,args.d).astype(np.float32)*2-1)
W1=0.1*(rs.rand(args.E,args.d,args.h).astype(np.float32)*2-1)
W2=0.1*(rs.rand(args.E,args.h,args.d).astype(np.float32)*2-1)
case=os.path.join(args.outdir,args.name); os.makedirs(case,exist_ok=True)
with open(os.path.join(case,"meta.txt"),"w") as f:
  f.write(f"N={args.N}\nB={args.B}\nd={args.d}\nh={args.h}\nE={args.E}\nE_local={E_local}\n")
write_bin(os.path.join(case,"Wg.bin"),Wg.reshape(-1),np.float32)
write_bin(os.path.join(case,"W1.bin"),W1.reshape(-1),np.float32)
write_bin(os.path.join(case,"W2.bin"),W2.reshape(-1),np.float32)
for r in range(args.N):
  X=(rs.rand(args.B,args.d).astype(np.float32)*2-1)
  scores=X@Wg.T
  top1=np.argmax(scores,axis=1).astype(np.int32)
  owner=(top1//E_local).astype(np.int32)
  exloc=(top1% E_local).astype(np.int32)
  Y=np.zeros((args.B,args.d),dtype=np.float32)
  for i in range(args.B):
    e=int(top1[i]); U=gelu(X[i]@W1[e]); Y[i]=U@W2[e]
  write_bin(os.path.join(case,f"X_rank{r}.bin"),X.reshape(-1),np.float32)
  write_bin(os.path.join(case,f"Y_rank{r}.bin"),Y.reshape(-1),np.float32)
  write_i32(os.path.join(case,f"top1_rank{r}.bin"),top1)
  write_i32(os.path.join(case,f"owner_rank{r}.bin"),owner)
  write_i32(os.path.join(case,f"exlocal_rank{r}.bin"),exloc)
print("wrote",case)
