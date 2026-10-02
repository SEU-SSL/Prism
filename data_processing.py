"""SCVHunter source adapter: deterministic token trees and CFG/DFG graphs."""
from pathlib import Path
import json, re, hashlib
import numpy as np
try:
 import dgl
except Exception: dgl=None

class Graph:
 def __init__(self,n,edges):
  self._n=n; self.edges=edges; self.ndata={}
 def num_nodes(self): return self._n
 def to(self,device):
  for k,v in self.ndata.items(): self.ndata[k]=v.to(device)
  return self

def make_graph(n, edges, feat, device=None):
 edges=list(edges)
 if dgl is not None:
  import torch
  g=dgl.graph(([a for a,b in edges],[b for a,b in edges]),num_nodes=n)
  g.ndata['w']=torch.as_tensor(feat,dtype=torch.float32,device=device)
  return g
 g=Graph(n,edges); import torch; g.ndata['w']=torch.as_tensor(feat,dtype=torch.float32,device=device); return g

def tokens(src):
 return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|=>|[{}()\[\];,.:+*/=<>-]',src)
def build_sample(path, label, dim=128):
 src=Path(path).read_text(errors='ignore'); ts=tokens(src) or ['empty']; n=min(max(len(ts),1),256); ts=ts[:n]
 ids=[int(hashlib.md5(t.encode()).hexdigest(),16)%dim for t in ts]
 feat=np.zeros((n,dim),dtype='float32'); feat[np.arange(n),ids]=1
 chain=[(i,i+1) for i in range(n-1)]
 cfg=make_graph(n,chain,feat); dfg=make_graph(n,chain+[(i,j) for i in range(n) for j in range(i+2,min(n,i+5))],feat)
 # BatchProgramClassifier tree: [token_id, child...], root first
 tree=[]
 for i,x in enumerate(ids): tree.append([x]+([i+1] if i+1<n else []))
 return {'path':str(path),'label':int(label),'tokens':tree,'cfg':cfg,'dfg':dfg,'ast_features':feat}
def discover(data,task,limit_per_class=None):
 root=Path(data)/task; groups=[]
 if task=='all':
  tasks=['reentrancy','origin','loop','blockinfo']
  for t in tasks: groups.extend(discover(data,t,limit_per_class))
  return groups
 dirs=[root/'dependency',root/'undependency'] if task!='blockinfo' else [p for p in root.iterdir() if p.is_dir()]
 for d in dirs:
  if not d.exists(): continue
  files=sorted(d.glob('*.sol')); files=files[:limit_per_class] if limit_per_class else files
  groups += [(f,1 if d.name=='dependency' else 0) for f in files]
 return groups
def prepare(data,task='all',limit_per_class=None,output=None):
 rows=[]
 for p,y in discover(data,task,limit_per_class):
  try: rows.append({'path':str(p),'label':y})
  except OSError: pass
 bundle={'data_root':str(data),'task':task,'samples':rows,'embedding_dim':128,'vocab_size':128}
 if output:
  out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); out.with_suffix('.json').write_text(json.dumps(bundle,indent=2))
 return bundle

def load_bundle(path): return json.loads(Path(path).read_text())
