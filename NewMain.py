"""Legacy source retained for reference; only the inline v4 implementation runs."""

# Original 889-line implementation (inactive, including its imports).
if False:
    import os
    import re
    import csv
    import dgl
    import datetime
    import json
    import numpy as np
    import solcx
    import solcast
    from anytree import AnyNode
    from glove import Corpus, Glove
    import torch as th
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    from gnnmodels.DistillationModel import Distillation
    from ParameterConfig import ParameterConfig
    from gensim.models import Word2Vec
    from evm_cfg_builder.cfg import CFG
    from ParameterConfig import ParameterConfig
    from solcx import compile_source, install_solc, set_solc_version
    FF = ParameterConfig.FF
    opcode_num = {}
    control = ['ForStatement','WhileStatement','IfStatement']
    Assignment = ['Assignment', 'BinaryOperation', 'UnaryOperation']
    MemberAccess = ['MemberAccess']
    PrimaryExpression = ['Identifier', 'ElementaryTypeName', 'VariableDeclaration', 'UserDefinedTypeName']
    NODE_NAME = ['FunctionDefinition', 'PrimaryExpression', 'VariableDeclaration']
    INST1 = ['LT','GT','SLT','SGT','EQ','ISZERO']
    INST2 = ['ADD', 'MUL', 'SUB', 'DIV', 'SDIV', 'SMOD', 'MOD', 'ADDMOD', 'MULMOD', 'EXP', 'SIGNEXTEND']
    INST3 = ['PUSH','POP','DUP','SWAP','INVALID']
    INST4 = ['AND','OR','XOR','NOT']
    INST5 = ['BYTE','SHL','SHR','SAR']
    INST6 = ['RETURNDATACOPY','SHA3','MLOAD','RETURN','LOG']#'MSIZE',
    INST7 = ['CALLDATALOAD','CALLDATACOPY','CODECOPY','EXTCODECOPY','RETURNDATACOPY','MSTORE','MSTORE8']
    INST8 = ['CREATE','CALL','CALLCODE','DELEGATECALL','CREATE2','STATICCALL','REVERT','MSIZE''SELFDESTRUCT','RETURNDATACOPY','SHA3','MLOAD',
    'RETURN','LOG','CALLDATALOAD','CALLDATACOPY','CODECOPY','EXTCODECOPY','RETURNDATACOPY','MSTORE','MSTORE8']
    INST9= ['BLOCKHASH','COINBASE','TIMESTAMP','NUMBER','DIFFICULTY','GASLIMIT','CHAINID','BASEFEE','ORIGIN','CALLVALUE','GASPRICE',
    'BALANCE','ADDRESS','SELFBALANCE','CALLDATALOAD','CALLDATACOPY','CALLDATASIZE','GAS','RETURNDATASIZE','RETURNDATACOPY','CODECOPY','CODESIZE',
    'EXTCODECOPY','EXTCODEHASH','EXTCODESIZE']
    F = ParameterConfig.viewfile
    device = th.device('cuda:0')
    tokens_dict = {}
    import json
    def get_standard_json(content):
        filepath = FF
        standard_json = '''
        {
            "language": "Solidity",
            "sources": {},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["*"]
                    },
                    "*": {
                        "": [ "ast" ]
                    }
                }
            }
        }
        '''
        standard_input = json.loads(standard_json)
        sources = {filepath: {'content': content}}
        standard_input["sources"] = sources
        return standard_input
    def Token_to_Emb(token,token_num):
        embsize = 128
        w2vec = Word2Vec(token, min_count=1, vector_size=embsize, workers=4)
        emb = np.zeros((token_num+1,embsize),dtype="float32")
        for i in range(token_num):
            emb[i] = w2vec.wv.get_vector(i)
        return emb
    def opop(op):
        opcode_to_id = {}
        current_id = 0
        for opcodes in op:
            for opcode in opcodes:
                if opcode not in opcode_to_id:
                    opcode_to_id[opcode] = current_id
                    current_id += 1
        opcodes_ids = []
        for opcodes in op:
            opcodes_ids.append([opcode_to_id[opcode] for opcode in opcodes])
        return opcodes_ids,opcode_to_id
    def byte_save(Opcodes,Sequences,CFGs_B,DFGs_B):# 格式处理
        fb =  file_load("embb.npy","dataset_B.json")
        opcodes_ids,opcode_to_id = opop(Opcodes)
        op = fb.view1s
        seqs_ids = []
        for seq in Sequences:
            seqs_ids.append([ opcode_to_id[s] for s in seq])
        CFG_s = fb.get_cfg()
        DFG_s = fb.get_dfg()
        emb_byte = fb.emb
        '''opcodes_ids,opcode_to_id = opop(Opcodes)
        cfgs_ids = []
        for cfg in CFGs_noP:'''
        return emb_byte,op,CFG_s,DFG_s
    class file_load:
        def __init__(self, embname, dataname):
            self.emb = np.load(F+embname)
            self.name = dataname
            self.labels = []
            self.view1s = []
            self.cfgs = []
            self.dfgs = []
            with open(F+dataname, 'r',encoding='UTF-8') as f:
                checkers = json.load(f)
                for checker in checkers:
                    self.labels.append(checker["label"])
                    self.view1s.append(checker["view1"])
                    self.cfgs.append(checker["view2"])
                    self.dfgs.append(checker["view3"])
        def get_ast(self):
            return self.view1s
        def get_cfg(self):
            gcs = []
            for cfg in self.cfgs:
                gc = dgl.graph((cfg[0]))
                gc = dgl.add_self_loop(gc)
                feature = []
                for node_id in gc.nodes():
                    feature.append(self.emb[cfg[1][int(node_id)]])
                gc.ndata['w'] = th.tensor(feature)
                gcs.append(gc)
            return gcs
        def get_dfg(self):
            gds = []
            for dfg in self.dfgs:
                gd = dgl.graph((dfg[0]))
                gd = dgl.add_self_loop(gd)
                feature = []
                for node_id in gd.nodes():
                    feature.append(self.emb[dfg[1][int(node_id)]])
                gd.ndata['w'] = th.tensor(feature)
                gds.append(gd)
            return gds
    
    def extract_version(contract_source):
        pragma_pattern = re.compile(r'pragma solidity (.*?);')
        match = pragma_pattern.search(contract_source)
        if match:
            version = match.group(1).strip()
            if version.startswith('^') or version.startswith('~'):
                return version[1:]
            return version
        return None
    import datetime
    class DFGBuilder:
        def __init__(self, key):
            self.key = key
            self.DFG_edges = []
            self.starttime = datetime.datetime.now().timestamp()
            self.lastwrite = -1
            self.lastwrite2 = -1
            self.lastwrite3 = -1
            self.lastwrite4 = -1
            self.memflag = set()
        def DFS_stack(self,blo,stack,i):
            for inst in blo.instructions:
                for _ in range(inst._pops):
                    try:
                        edge = [stack.pop(),inst._pc]
                        if  edge not in self.DFG_edges:
                            self.DFG_edges.append(edge)
                    except:
                        "ERROR:stack empty!"
                if inst._pushes!=0:
                    num = inst._pushes
                    stack.extend([inst._pc for _ in range(num)])
            for child in sorted(blo.outgoing_basic_blocks(self.key), key=lambda x:x.start.pc):
                if "flag" not in child._incoming_basic_blocks:
                    child._incoming_basic_blocks["flag"] = [stack[-1:]]
                    self.DFS_stack(child,[s for s in stack])
                elif stack[-1:] not in child._incoming_basic_blocks["flag"]:
                    child._incoming_basic_blocks["flag"].append(stack[-1:])
                    self.DFS_stack(child,[s for s in stack[-20:]])
        def build_dfg(self,blo,):
            self.DFS_stack(blo,[])
            return self.DFG_edges
        def DFS_stack(self,blo,stack,i,j):
            mem = []
            for inst in blo.instructions:
                if inst._name in INST9: 
                    if self.lastwrite3 !=-1:
                        mem.append([self.lastwrite3,inst._pc])
                    self.lastwrite3 = inst._pc
                if inst._name in INST8: 
                    if self.lastwrite !=-1:
                        mem.append([self.lastwrite,inst._pc])
                    self.lastwrite = inst._pc
                    '''if inst._name in INST_mem_write: 
                        if self.lastwrite !=-1:
                            mem.append([self.lastwrite,inst._pc])
                        self.lastwrite = inst._pc
                    elif self.lastwrite !=-1 and inst._name in INST_mem_read:
                        mem.append([self.lastwrite,inst._pc])'''
                elif inst._name in ['SLOAD','SSTORE']:
                    if self.lastwrite2 !=-1:
                        mem.append([self.lastwrite2,inst._pc])
                    self.lastwrite2 = inst._pc
                if inst._name == 'DUP':
                    num = inst._pushes - 1
                    if len(stack)>=num:
                        stack.append(stack[-num])
                    else:
                        "ERROR:stack empty!DUP"
                elif inst._name == 'SWAP':
                    try:
                        num = inst._pops
                        i = stack[-num]
                        stack[-num] = stack[-1]
                        stack[-1] = i
                    except:
                        "ERROR:stack empty!SWAP"
                else:
                    for _ in range(inst._pops):
                        try:
                            edge = [stack.pop(),inst._pc]
                            if  edge not in self.DFG_edges:
                                self.DFG_edges.append(edge)
                        except:
                            "ERROR:stack empty!"
                    if inst._pushes!=0:
                        num = inst._pushes
                        stack.extend([inst._pc for _ in range(num)])
            if blo.instructions[0]._pc not in self.memflag:
                self.DFG_edges += mem
                self.memflag.add(blo.instructions[0]._pc)
            for child in sorted(blo.outgoing_basic_blocks(self.key), key=lambda x:x.start.pc):
                if "flag" not in child._incoming_basic_blocks:
                    child._incoming_basic_blocks["flag"] = [stack[-1:]]
                    self.DFS_stack(child,[s for s in stack])
                elif stack[-1:] not in child._incoming_basic_blocks["flag"]:
                    child._incoming_basic_blocks["flag"].append(stack[-1:])
                    self.DFS_stack(child,[s for s in stack[-20:]])
        def DFS_stack(self,blo,stack):
            mem = []
            for inst in blo.instructions:
                if inst._name in INST7:
                    if self.lastwrite !=-1:
                        mem.append([self.lastwrite,inst._pc])
                    self.lastwrite = inst._pc
                elif self.lastwrite !=-1 and inst._name in INST6:
                    mem.append([self.lastwrite,inst._pc])
                elif inst._name in ['SLOAD','SSTORE']:
                    if self.lastwrite2 !=-1:
                        mem.append([self.lastwrite2,inst._pc])
                    self.lastwrite2 = inst._pc
                if inst._name == 'DUP':
                    num = inst._pushes - 1
                    if len(stack)>=num:
                        stack.append(stack[-num])
                    else:
                        "ERROR:stack empty!DUP"
                elif inst._name == 'SWAP':
                    try:
                        num = inst._pops
                        i = stack[-num]
                        stack[-num] = stack[-1]
                        stack[-1] = i
                    except:
                        "ERROR:stack empty!SWAP"
                else:
                    for _ in range(inst._pops):
                        try:
                            edge = [stack.pop(),inst._pc]
                            if  edge not in self.DFG_edges:
                                self.DFG_edges.append(edge)
                        except:
                            "ERROR:stack empty!"
                    if inst._pushes!=0:
                        num = inst._pushes
                        stack.extend([inst._pc for _ in range(num)])
            if blo.instructions[0]._pc not in self.memflag:
                self.DFG_edges += mem
                self.memflag.add(blo.instructions[0]._pc)
            for child in sorted(blo.outgoing_basic_blocks(self.key), key=lambda x:x.start.pc):
                if "flag" not in child._incoming_basic_blocks:
                    child._incoming_basic_blocks["flag"] = [stack[-1:]]
                    self.DFS_stack(child,[s for s in stack])
                elif stack[-1:] not in child._incoming_basic_blocks["flag"]:
                    child._incoming_basic_blocks["flag"].append(stack[-1:])
                    self.DFS_stack(child,[s for s in stack[-20:]])
    with open(FF+'opcode.txt', 'r',encoding='UTF-8') as file:
        for index, line in enumerate(file):
            opcode = line.strip()
            opcode_num[opcode] = index
    class GloveUtils:
        def TrainGlove(self):
            f = csv.reader(open(self.train_data))
            all_texts = []
            for lst in f:
                all_texts.append(lst[1].split())
            corpus = Corpus()
            corpus.fit(all_texts, window=self.glove_window)
            glove = Glove(no_components=self.components, learning_rate=self.glove_learning_rate)
            glove.fit(corpus.matrix, epochs=self.glove_epoch, no_threads=self.workers, verbose=True)
            glove.add_dictionary(corpus.dictionary)
    def Sourcecode_to_Bytecode(sourcecode):
        version = extract_version(sourcecode)
        if version==None:
            return None
        install_solc(version)
        set_solc_version(version)
        compiled_sol = compile_source(sourcecode)
        if 'errors' in compiled_sol:
            return None
        bytecode = "".join(contract_data.get("bin-runtime", "") for contract_data in compiled_sol.values())
        if len(bytecode)==0:
            return None
        return '0x'+bytecode
    
    def Member_function(node):
        for child in node._children:
            if child.nodeType == 'MemberAccess':
                return True
            if Member_function(child):
                return True
        return False
    def delegate_function(node,withdraw_function,CG_list):
        for child in node._children:
            try:
                if child.name in withdraw_function:
                    CG_list.append([child.id,withdraw_function[child.name]])
            except:
                delegate_function(child,withdraw_function,CG_list)
    def get_Member_functions(pruned_ast):
        # 用于生成CFG的函数调用关系
        Members = []
        CG_lists = {}
        source_node = set()
        num = 1
        for child in pruned_ast._children:
            key = 0
            if child.nodeType == 'ContractDefinition':
                contract_node = set()
                withdraw_function = {}
                other_function_list = []
    
                for node in child:
                    if node.nodeType == 'FunctionDefinition':
                        node_useful = set()
                        for node_c in node._children:
                            if len(node_c._children) > 0:
                                node_useful.add(node_c)
                        node._children = node_useful
                        if Member_function(node):
                            key = 1
                            contract_node.add(node)
                            withdraw_function[node.name] = node.id
                            Members.append(node.id)#控制流处理
                        else:
                            other_function_list.append(node.name)
                for node in child:
                    if node.nodeType == 'FunctionDefinition':
                        if node.name in other_function_list:
                            CG_list = []
                            delegate_function(node,withdraw_function,CG_list)
                            if len(CG_list)>0:
                                contract_node.add(node)
                                CG_lists[node.id] = CG_list
                    else:
                        if len(node._children) > 0:
                            contract_node.add(node)
                child._children = contract_node
                if key == 1:
                    source_node.add(child)
            else:
                source_node.add(child)
        pruned_ast._children = source_node
        return pruned_ast,CG_lists,Members
    def next_declaration(pruned_ast, token):
        for child in pruned_ast._children:
            try:
                if child.name in token:
                    child.name = token.get(child.name)
            except:
                True
            next_declaration(child, token)
    def find_declaration(source_ast, token, id):
        for child in source_ast._children:
            try:
                if child.name in token:
                    child.name = token.get(child.name)
                elif child.nodeType == 'VariableDeclaration':
                    token[child.name] = "VAR_"+str(id["v"])
                    child.name = "VAR_"+str(id["v"])
                    id["v"] = id["v"]+1
                elif child.nodeType == 'FunctionDefinition':
                    token[child.name] = "FUNC_"+str(id["f"])
                    child.name = "FUNC_"+str(id["f"])
                    id["f"] = id["f"]+1
                elif child.nodeType == 'EventDefinition':
                    token[child.name] = "EVENT_"+str(id["e"])
                    child.name = "EVENT_"+str(id["e"])
                    id["e"] = id["e"]+1
                elif child.nodeType == 'ContractDefinition' or child.nodeType == 'UserDefinedTypeName':
                    token[child.name] = "CON_"+str(id["c"])
                    child.name = "CON_"+str(id["c"])
                    id["c"] = id["c"]+1
                elif child.nodeType == 'ElementaryTypeName':
                    child.name = 'ElementaryTypeName'
            except:
                try:
                    if child.nodeType in Assignment:
                        child.operator = child.nodeType
                except:
                    True
            find_declaration(child,token, id)
    def simple_ast(source_ast):
        id = {"v":0,"f":0,"e":0,"c":0}
        token = {}
        find_declaration(source_ast, token, id)
        next_declaration(source_ast, token)  
    
    def digui(pruned_ast,i):
        for child in pruned_ast._children:
            print("      "*i,child.id,child)
            digui(child,i+1)
    def create_tree(dict_trans,root, node, node_list, parent=None):
        id = len(node_list)
        token, children = get_token(node), get_children(node)
        num = tokens_dict[token]
        dict_trans[node.id] = [id,num]
        if id == 0:
            root.token = token
            root.data = node
        else:
            new_node = AnyNode(id=id, token=token, data=node, parent=parent)
        node_list.append(node)
        tokens = []
        tokens.append(num)
        for child in children:
            if id == 0:
                tokens.append(create_tree(dict_trans, root, child, node_list, parent=root))
            else:
                tokens.append(create_tree(dict_trans, root, child, node_list, parent=new_node))
        return tokens
    def get_token(node):
        if hasattr(node, 'nodeType'):
            if node.nodeType == 'FunctionDefinition':
                if node.name == '':
                    return 'payable'
                return node.name
            if node.nodeType in NODE_NAME:
                return node.name
            if node.nodeType in PrimaryExpression:
                return node.name
            if node.nodeType in MemberAccess:
                return node.memberName
            if node.nodeType in Assignment:
                return node.operator
            if node.nodeType is not None:
                return node.nodeType
        return None
    def get_children(node):
        children = node._children
        return children
    def get_sequence(node, tokens):
        token = get_token(node)
        children = get_children(node)
        if token is not None:
            tokens.append(token)
        for child in children:
            get_sequence(child, tokens)
    def node_index_list_generate(node, node_index_list):
        token = node.token
        node_index_list.append(tokens_dict[token])
        for child in node.children:
            node_index_list_generate(child, node_index_list)
    def add_control_flow(node,tree_now):
        tree_now.append(node.data.id)
        for child in node.children:
            add_control_flow(child,tree_now)
    def F_control_flow(node, tree_now, tree_all):
        flag = 0
        control = ['ForStatement','WhileStatement','IfStatement']
        if node.data.nodeType == "FunctionCall":
            for child in node.children:
                if child.data.name == "require":
                    flag = 1
                    break
            if flag == 1:
                add_control_flow(node,tree_now)
                return 2
        elif node.data.nodeType in control:
            tree_now.append(node.data.id)
            condition = node.data.condition  # 判断节点
            for child in node.children:
                if child.data.id == condition.id:
                    add_control_flow(child,tree_now)
                    break
        for child in node.children:
            if child.data.nodeType == 'MemberAccess':
                add_control_flow(child,tree_now)
                tree_all.append(tree_now)
            else:
                tr = [i for i in tree_now]
                flag = F_control_flow(child, tree_now, tree_all)
                if flag == 0:
                    tree_now = tr
                else: flag = flag -1
        return flag
    def W_control_flow(m_list, node, tree_now, tree_all):
        if node.data.nodeType in control:
            tree_now.append(node.data.id)
            condition = node.data.condition
            for child in node.children:
                if child.data.id == condition.id:
                    add_control_flow(child,tree_now)
                    break
        for child in node.children:
            if child.data.id in m_list:
                tree_now.append(child.data.id)
                tree_all.append(tree_now)
            else:
                tr = [i for i in tree_now]
                W_control_flow(m_list, child, tree_now, tree_all)
                tree_now = tr
    def Member_control_flow(CG_list, Member, node, tree_all):
        for contract in node.children:
            if contract.token == 'ContractDefinition':
                for func in contract.children:
                    if func.id in CG_list:
                        tree_now = [contract.data.id, func.data.id]
                        W_control_flow(CG_list[func.id], func, tree_now, tree_all)
                    elif func.id in Member:
                        tree_now = [contract.data.id, func.data.id]
                        F_control_flow(func, tree_now, tree_all)
        return tree_all
    def add_node1(node, token,i):
        for child in node.children:
            if child.data.nodeType in ['Identifier']:
                if 'FUNC' in child.data.name:
                    continue
                elif child.data.name in token:
                    token[child.data.name][0].append(child.data.id)
                else:
                    token[child.data.name] = [[child.data.id],[],[]]
            elif i==0 and child.data.nodeType not in ['MemberAccess','BinaryOperation','IndexAccess']:
                continue
            add_node1(child, token, 1)
    def add_node2(node, token,ll):
        if node.data.nodeType in ['Identifier']:
            if 'FUNC' not in node.data.name:
                for l in ll:
                    token[l.data.name][2].append([l.data.id,node.data.id])
        for child in node.children:
            add_node2(child, token, ll)
    def find_token(node, token, ll):
        if (node.data.nodeType == 'Identifier') and (node.data.name in token) :#and (node.id < max(token[node.data.name][0]))
            ll.append(node)
        for child in node.children:
            find_token(child, token, ll)
    def value_computed_from(node, token):
        for child in node.children:
            if child.data.nodeType == 'Assignment':
                ll = []
                for chch in child.children:
                    if chch.data.id == child.data.leftHandSide.id:
                        find_token(chch, token,ll)
                        break
                if len(ll)>0:
                    for chch in child.children:
                        if chch.data.id == child.data.rightHandSide.id:
                            add_node2(chch, token, ll)
                            break
            elif child.data.nodeType == 'VariableDeclarationStatement':
                if len(child.data.assignments) > 0:
                    ll = []
                    for chch in child.children:
                        if (chch.data.id == child.data.declarations[0].id) and (chch.data.name in token):
                            ll.append(chch)
                            break
                    if len(ll)>0:
                        for chch in child.children:
                            if (chch.data.id == child.data.initialValue.id):
                                add_node2(chch, token, ll)
                                break
            else:
                value_computed_from(child, token)
    def value_comes_from(node, token):
        for child in node.children:
            try:
                if (child.data.name in token) and (child.data.id not in token[child.data.name][0]):
                    token[child.data.name][1].append(child.data.id)
            except:
                value_comes_from(child, token)
    def get_variables_nodes(node, token):
        for child in node.children:
            if child.data.nodeType == 'MemberAccess':
                add_node1(node, token,0)
                continue
            get_variables_nodes(child, token)
    
    def DataflowSet():
        "具体内容参考文件：generate_graph_mvl2.py"
        '''get_leaf_node_edge(new_tree, tokens_dict, edge_d_source, edge_d_target, edge_d_type)
        # 获取 AST 中变量节点
        variables_token = get_variables_token(tree)
        # 生成变量之间的边
        get_variables_node_edge(new_tree, tokens_dict, edge_d_source, edge_d_target, edge_d_type, variables_token)
        
        dict_trans_trans = {}
        def digui3(node,dict_trans_trans):
            dict_trans_trans[node.id] = node.data.id
            for child in node.children:
                digui3(child, dict_trans_trans)
        digui3(new_tree, dict_trans_trans)
        for j in range(len(edge_d_source)):
            edge_d_source[j] = dict_trans_trans[edge_d_source[j]]
            edge_d_target[j] = dict_trans_trans[edge_d_target[j]]
    
        edge_data = [edge_d_source, edge_d_target]
        n_trans(edge_data, node_trans_d)
        edge_d_source = [node_trans_d[i] for i in edge_data[0]]
        edge_d_target = [node_trans_d[i] for i in edge_data[1]]'''
    def Member_data_flow(Member, node, edge_data):  
        # 忘记具体实验用的哪个版本，如果这个效果不好试试函数DataflowSet
        '''
        Member.append(node.id)
        '''
        for contract in node.children:
            if contract.token == 'ContractDefinition':
                for func in contract.children:
                    if func.id in Member:
                        variables_token = {}
                        get_variables_nodes(func,variables_token)
                        value_comes_from(func,variables_token)
                        value_computed_from(func,variables_token)
                        for token in variables_token:
                            key = variables_token[token][0][0]
                            ll = (variables_token[token][0]+variables_token[token][1])
                            ll.sort()
                            if len(ll) >= 2:
                                for i in range(len(ll)-1):
                                    edge_data.append([ll[i+1],ll[i]])
                                    '''if ll[i] == key:
                                        key = -key
                                    if key < 0:
                                        edge_data.append([ll[i+1],ll[i]])
                                    else:
                                        edge_data.append([ll[i],ll[i+1]])'''
                                        #edge_data[0].append(ll[i])
                                        #edge_data[1].append(ll[i+1])
                            for t in variables_token[token][2]:
                                edge_data.append(t)
    def cg_trans(CG_list, dict_trans, CG_list_trans, Member, Member_trans, tree_all):
        for node in CG_list:
            new_list = []
            id_s = dict_trans[node][0]
            for i in CG_list[node]:
                id_t = dict_trans[i[1]][0]
                new_list.append(i[0])
                tree_all.append([node,i[1]])
            CG_list_trans[dict_trans[node][0]] = new_list
        for node in Member:
            Member_trans.append(dict_trans[node][0])
    def n_trans(tree_all,node_trans):
        node = set()
        for a in tree_all:
            node.update(a)
        i = 0
        for b in node:
            node_trans[b] = i
            i = i + 1
    def graph_generate(node_trans, tree_all, edge_source, edge_target):
        graph = {}
        for tree in tree_all:
            for i in range(len(tree)-1):
                if tree[i] in graph:
                    graph[tree[i]].update([tree[i+1]])
                else: 
                    graph[tree[i]] = set([tree[i+1]])
        for source in graph:
            for target in graph[source]:
                edge_source.append(node_trans[source])
                edge_target.append(node_trans[target])
    def pad_nodes(nodes_attrs):
        torches = []
        for node_atts in nodes_attrs:
            tmp_torch = th.tensor(node_atts)
            actual_len = len(tmp_torch)
            if actual_len > 20:
                tmp_torch = tmp_torch[:20]
            else:
                tmp_torch = th.cat([tmp_torch, tmp_torch.new_zeros(20 - actual_len)], 0)
            torches.append(tmp_torch)
        return th.stack(torches, 0)
    def create_separate_graph(tree, CG_list, Member):
        node_list = []
        dict_trans = {}
        new_tree = AnyNode(id=0, token=None, data=None)
        token = []
        edge_c_source = []
        edge_c_target = []
        edge_d_source = []
        edge_d_target = []
        CG_list_trans = {}
        Member_trans = [] 
        tree_all = []
        edge_data = []
        node_trans_c = {}
        node_trans_d = {}
        ast_nodes = create_tree(dict_trans, new_tree, tree, node_list)
        cg_trans(CG_list, dict_trans, CG_list_trans, Member, Member_trans, tree_all)
        node_index_list_generate(new_tree, token)
        Member_control_flow(CG_list_trans, Member_trans, new_tree, tree_all)
        n_trans(tree_all,node_trans_c)
        graph_generate(node_trans_c, tree_all, edge_c_source, edge_c_target)
        Member_data_flow(Member_trans, new_tree, edge_data)
        n_trans(edge_data,node_trans_d)
        graph_generate(node_trans_d, edge_data, edge_d_source, edge_d_target)
        if len(edge_c_source) <= 1:
            return [],0,0,0
            #gc = [[[0,0]],[[0]]]
        else:
            lis = []
            for i in range(len(edge_c_source)):
                lis.append([edge_c_source[i], edge_c_target[i]])
            gc = [lis,[dict_trans[i][1] for i in node_trans_c]]
        
        if len(edge_d_source) <= 1 :
            return [],0,0,0
            #gd = [[[0,0]],[[0]]]
        else:
            lis = []
            for i in range(len(edge_d_source)):
                lis.append([edge_d_source[i], edge_d_target[i]])
            gd = [lis,[dict_trans[i][1] for i in node_trans_d]]
        return token,ast_nodes,gc,gd
    def Sourcecode_to_ACD(sourcecode):
        input_json = get_standard_json(sourcecode)
        try:
            output_json = solcx.compile_standard(input_json)
        except:
            print("ast error")
            return []
        source_nodes = solcast.from_standard_output(output_json)
        token = []
        for source_ast in source_nodes:
            simple_ast(source_ast)
            get_sequence(source_ast, token)
            source_ast,CG_list,Member = get_Member_functions(source_ast)
            ast = source_ast
        unique_token = []
        for item in token:
            if item not in unique_token:
                unique_token.append(item)
        new_unique_tokens = [t for t in unique_token if t not in tokens_dict]
        max_id = max(tokens_dict.values()) if tokens_dict else -1
        new_ids = range(max_id + 1, max_id + 1 + len(new_unique_tokens))
        tokens_dict.update(dict(zip(new_unique_tokens, new_ids)))
        return create_separate_graph(ast, CG_list, Member)
    def Bytecode_to_OCD(sourcecode):
        try:
            bytecode = Sourcecode_to_Bytecode(sourcecode)
        except:
            return [],0,0,0
        if bytecode == None:
            return [],0,0,0
        cfg = CFG(bytecode)
        num_to_inst = dict()
        pc_to_num = dict()
        DFG_edges = []
        CFG_edges = []
        opcode = []
        sequence = []
        i = 0
        for inst in cfg._instructions:
            instName = cfg._instructions[inst]._name
            if instName in INST2:
                instName = 'Arith'
            elif instName in INST4:
                instName = 'Bit'
            elif instName in INST5:
                instName = 'INS3'
            elif instName in INST1:
                instName = 'Logic'
            opcode.append(opcode_num[instName])
            print(instName)
            pc_to_num[cfg._instructions[inst]._pc] = i
            num_to_inst[i] = opcode_num[instName]
            i+=1
            if instName not in INST3:
                sequence.append(opcode_num[instName])
    
        for function in sorted(cfg.functions, key=lambda x: x.start_addr):
            for basic_block in sorted(function.basic_blocks, key=lambda x: x.start.pc):
                inst_front = basic_block.instructions[0]
                for inst in basic_block.instructions[1:]:
                    if inst_front._name in INST3:
                        inst_front = inst
                        continue
                    elif inst._name not in INST3:
                        CFG_edges.append( [pc_to_num[inst_front._pc],pc_to_num[inst._pc]] )
                        inst_front = inst
                if inst_front._name not in INST3:
                    for outgoing_bb in sorted( basic_block.outgoing_basic_blocks(function.key), key=lambda x: x.start.pc ):
                        for inst_out in outgoing_bb.instructions:
                            if inst_out._name not in INST3:
                                CFG_edges.append( [pc_to_num[inst_front._pc],pc_to_num[inst_out._pc]] )
                                break
                '''inst_front = basic_block.instructions[0] #back
                for inst in basic_block.instructions[1:]:
                    CFG_nop.append( [pc_to_num[inst_front._pc],pc_to_num[inst._pc]] )
                    inst_front = inst
                for outgoing_bb in sorted( basic_block.outgoing_basic_blocks(function.key), key=lambda x: x.start.pc ):
                    CFG_edges.append( [pc_to_num[inst_front._pc],pc_to_num[outgoing_bb.instructions[0]._pc]] )'''
                if len( basic_block.incoming_basic_blocks(function.key) )==0 and 'flag' not in basic_block._incoming_basic_blocks:
                    dfg_builder = DFGBuilder(function.key)#(blo_to_key)
                    DFG_edges += dfg_builder.build_dfg(basic_block)
        if len(CFG_edges)<=1 or len(DFG_edges)<=1:
            return [],0,0,0
        mapping_dict = {}
        current_index = 0
        DEG_new_list = [[ mapping_dict.setdefault(num, current_index + len(mapping_dict)) for num in sublist ] for sublist in DFG_edges]
        dnum_to_inst = { new_num: num_to_inst[pc_to_num[pc]] for pc, new_num in mapping_dict.items() }
        DEG_new_list = [[ pc_to_num[num] for num in sublist ] for sublist in DFG_edges]
        return opcode,sequence,[CFG_edges,num_to_inst],[DEG_new_list,num_to_inst]
    def Datab(name):
        Opcodes = []
        CFGs_B = []
        DFGs_B = []
        Sequences = []
        for root1, dirs1, files1 in os.walk(FF+name):
            for contract_type in dirs1:
                for root2, dirs2, files in os.walk(os.path.join(FF+name, contract_type)):
                    for file in files:
                        filepath = os.path.join(root2, file)
                        try:
                            print("数据处理")
                            with open(filepath, encoding="utf-8") as f:
                                sourcecode = f.read()
                                opcode,sequence,cdb,ddb= Bytecode_to_OCD(sourcecode)
                                if len(opcode)<10:
                                    continue
                                Sourcecode_to_ACD(sourcecode)
                                Opcodes.append(opcode)
                                Sequences.append(sequence)
                                CFGs_B.append(cdb)
                                DFGs_B.append(ddb)
                                #CFGs_noP.append(cdp)
                                #DFGs_noP.append(ddp)
                        except:
                            1
        return Opcodes,Sequences,CFGs_B,DFGs_B
    
    def Model_(Opcodes,Sequences,CFGs_B,DFGs_B):
        print("数据载入")
        def get_seq(Opcodes,emb_byte):
            max_len = 8000
            vectorized_nodes = []
            for opcode_seq in Opcodes:
                vec_sequence = []
                for opcode in opcode_seq:
                    opcode_vector = emb_byte[opcode]
                    vec_sequence.append(opcode_vector)
                if len(vec_sequence) < max_len:
                    padding = [np.zeros(emb_byte.shape[1]) for _ in range(max_len - len(vec_sequence))]
                    padded_seq = padding + vec_sequence
                else:
                    padded_seq = vec_sequence[:max_len]
                vectorized_nodes.append(padded_seq)
            return np.array(vectorized_nodes)
        fs =  file_load("embs.npy","dataset_S.json")
        emb_source = fs.emb
        Labels = fs.labels
        emb_byte,opcodes_ids,CFG_b,DFG_b =  byte_save(Opcodes,Sequences,CFGs_B,DFGs_B)
        Opcodes = get_seq(opcodes_ids,emb_byte)
        Asts = fs.get_ast()
        CFG_s = fs.get_cfg()
        DFG_s = fs.get_dfg()
        dataset = list(zip(Labels,Opcodes,CFG_b,DFG_b,Asts,CFG_s,DFG_s))
        print(len(Labels),len(Opcodes),len(CFG_b),len(DFG_b),len(Asts),len(CFG_s),len(DFG_s))
        train_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=ParameterConfig.dataset_split_ratio, stratify=Labels, random_state=1
        )
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        model = Distillation(emb_source,emb_byte)
        print("模型载入")
        model.model_(train_dataset, test_dataset)
        #model.no_D(train_dataset, test_dataset)
    
    if __name__ == '__main__':
        Opcodes,Sequences,CFGs_B,DFGs_B = Datab("dataset/timestamp")
        Model_(Opcodes,Sequences,CFGs_B,DFGs_B)


# Active v4 implementation, inlined below (no entry-point forwarding).
import argparse,json,random
from pathlib import Path
import numpy as np, torch
from ParameterConfig import ParameterConfig
from data_processing import prepare as prep, load_bundle, build_sample
from models.S_Model import S_Model

def main():
 p=argparse.ArgumentParser(description='Prism-v4 SCVHunter source GNN'); sub=p.add_subparsers(dest='cmd',required=True)
 q=sub.add_parser('prepare'); q.add_argument('--data',default='dataset/scvhunter'); q.add_argument('--task',default='all'); q.add_argument('--output',default='artifacts/prepared'); q.add_argument('--limit-per-class',type=int)
 t=sub.add_parser('train'); t.add_argument('--prepared',required=True); t.add_argument('--task'); t.add_argument('--device',choices=['cpu','cuda','auto'],default='auto'); t.add_argument('--epochs',type=int,default=1); t.add_argument('--output',default='artifacts')
 a=p.parse_args()
 if a.cmd=='prepare':
  b=prep(a.data,a.task,a.limit_per_class,a.output); print(json.dumps({'samples':len(b['samples']),'output':str(Path(a.output).with_suffix('.json'))})); return
 b=load_bundle(a.prepared); dev='cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu'; rows=b['samples']; random.Random(1).shuffle(rows)
 emb=np.eye(128,dtype='float32'); model=S_Model(128,128,ParameterConfig.HEAD_NUM,emb,2,128,emb,device=dev).to(dev); opt=torch.optim.Adam(model.parameters(),lr=ParameterConfig.lr); lossfn=torch.nn.CrossEntropyLoss(); outputs=[]
 for ep in range(a.epochs):
  for r in rows:
   s=build_sample(r['path'],r['label']); y=torch.tensor([r['label']],device=dev); out,_=model([s['tokens']],s['cfg'].to(dev),s['dfg'].to(dev),[r['label']],ep); loss=sum(lossfn(x,y) for x in out); opt.zero_grad(); loss.backward(); opt.step(); outputs.append(float(loss))
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/'model.pt'); (out/'training.json').write_text(json.dumps({'task':b.get('task'),'device':dev,'epochs':a.epochs,'samples':len(rows),'outputs':7,'last_loss':outputs[-1] if outputs else None})); print(json.dumps({'device':dev,'samples':len(rows),'outputs':7}))
if __name__=='__main__': main()
