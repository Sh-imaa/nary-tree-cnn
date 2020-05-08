import os
import re
import random
import copy
from collections import defaultdict

import pandas as pd


def get_sentences(lines):
    new_sentence = False
    tree = []
    sentence_tree = []
    for line in lines:
        if (line[:8] == 'Sentence' )and new_sentence:
            s = ' '.join(sentence_tree)
            s = re.sub('\s+', ' ', s)
            s = re.sub('\(', '( ', s)
            s = re.sub('\)', ' )', s)
            tree.append(re.sub('\s+', ' ', s))
            sentence_tree = []
            new_sentence = False
        if new_sentence:
            sentence_tree.append(line)
        if line == 'Constituency parse: \n':
            new_sentence = True
            
    s = ' '.join(sentence_tree)
    s = re.sub('\s+', ' ', s)
    s = re.sub('\(', '( ', s)
    s = re.sub('\)', ' )', s)
    tree.append(re.sub('\s+', ' ', s))

    return tree


class Node:
    def __init__(self, POS, word=None):
        self.word = word
        self.c = []
        self.isLeaf = False
        self.isRoot = False
        self.POS = POS
        self.level = 0
        self.max_nodes = 0


class Tree:
    def __init__(self, treeStrings, label):
        self.label = label
        self.open = '('
        self.close = ')'
        self.root = Node('Main ROOT')
        self.root.isRoot = True
        for treeString in treeStrings:
            self.tokens = treeString.strip().split()
            self.parse()

    def get_words(self):
        nodes_list = []
        traverse(self.root,  lambda node, args: args.append(node), [nodes_list])
        return [node.word for node in nodes_list if node.isLeaf]

        
    def parse(self):
        parents = []
        count = 0
        node_info = [None, None]
        info_position = 0
        last_close = False
        start = True
        for token in self.tokens:
            assert count >= 0, "Malformed tree"
            
            if token == '(':
                if info_position == 1:
                    node = Node(node_info[0])
                    
                    if start:
                        self.root.c.append(node)
                        start = False
                    else:
                        parents[-1].c.append(node)
                    parents.append(node)
                node_info = [None, None]
                info_position = 0
                count += 1
                last_close = False
            
            elif token == ')':
                if info_position == 2:
                    node = Node(node_info[0], node_info[1])
                    node.isLeaf = True
                    parents[-1].c.append(node)
                elif last_close:
                    parents.pop()

                count -= 1
                last_close = True
                info_position = 0 
            
            else:
                node_info[info_position] = token
                info_position += 1
                last_close = False
                
        assert count == 0, "Malformed tree"
                
def traverse(node, nodeFn=None, args=None):
    for child in node.c:
        traverse(child, nodeFn, args)
    nodeFn(node, *args)

def shrink(node, parent, j=0):
    if not node.c:
        return
    if len(node.c) == 1:
        if parent is not None:
            parent.c[j] = node.c[0]
        else:
            node.isLeaf = node.c[0].isLeaf
            node.word = node.c[0].word
            node.c = node.c[0].c
            node.isRoot = True
            shrink(node, None, 0)
    for i, n in enumerate(node.c):
        shrink(n, node, i)

def get_nodes_per_level(node, node_by_level):
    node_by_level[node.level].append(node)
    for n in node.c:
        get_nodes_per_level(n, node_by_level)

def get_max_nodes(node, max_nodes):
    node.max_nodes = max_nodes[node.level]
    for n in node.c:
        get_max_nodes(n, max_nodes)

def get_first_tree(tree):
    short_tree = copy.deepcopy(tree)
    try:
        if short_tree.root.c[0].POS == 'S':
            short_tree.root = short_tree.root.c[0]
            short_tree.root.isRoot = True
        return short_tree
    except:
        return short_tree


def generate_levels(node):
    if node.isLeaf:
        node.level = 0
        return node.level
    levels = []
    for n in node.c:
        levels.append(generate_levels(n))
    node.level = max(levels) + 1
    return node.level

def pad(node):
    if node.isLeaf:
        return
    pad_length = node.max_nodes - len(node.c)
    if pad_length > 0:
        n = Node('extra', word='<pad>')
        n.isLeaf = True
        node.c.extend([n]*pad_length)
    for n in node.c:
        pad(n)

def load_shrinked_trees(trees_path, data_path):
    print('loading trees ....')
    df = pd.read_csv(data_path)
    trees = []
    for folder in os.listdir(trees_path):
        tree_folder = os.path.join(trees_path, folder)
        if  os.listdir(tree_folder):
            tree_path = os.path.join(tree_folder, os.listdir(tree_folder)[0])

            with open(tree_path) as f:
                lines = f.readlines()
                tree_list = get_sentences(lines)
                # fix nan
                df.loc[(df.polarity != 1), 'polarity'] = [0] * len(df[df.polarity != 1])
                t = Tree(tree_list, df[df.id == int(folder)].polarity.values[0])
                shrink(t.root, None)
                generate_levels(t.root)
                trees.append(t)

    print('trees loaded')
    return trees

def get_height(node, level=0):
    if node.isLeaf:
        return level
    return max([get_height(n, level + 1) for n in node.c])

def print_node(node, prefix=''):
    print(prefix + str(node.word))
    prefix = prefix + '  '
    for c in node.c:
        print_node(c, prefix)
    

def edit_tree(tree, edits=1):
    h = get_height(tree.root)
    if h <= 2:
        return 0 
    edit_count = [0]
    edited_levels = []
    # pick any valid level to edit
    while  (edit_count[0] < edits) and (len(edited_levels) < (h - 2)):
        level = random.randrange(2, h)
        if level in edited_levels:
            continue
        edited_levels.append(level)

        traverse(tree.root, edit_level, [level, edits, edit_count])

    generate_levels(tree.root)
    return edit_count[0]

def edit_level(node, level, max_edits=1, edits_c=[0]):
    if (edits_c[0] < max_edits) and (node.level == level):
        if edit_node(node): edits_c[0] += 1

def edit_node(node):
    if node.level < 2:
        return node
    all_c = []
    old_order = []

    for c in node.c:
        if c.isLeaf:
            all_c.append(c)
            old_order.append(1)
        else:
            all_c += c.c
            old_order.append(len(c.c))
    n = len(all_c)
    # print('old order', old_order)
    # print('old nodes', node.c)
    new_c = []
    new_order = []
    while (n > 2):
        m = random.randrange(1, n, 1)
        # print('m: ', m)
        new_order.append(m)
        if m == 1:
            new_c.append(all_c.pop())
        else:
            new_n = Node('edited')
            new_n.c = all_c[:m]
            del all_c[:m]
            new_c.append(new_n)
        n = n - m
    # add what left
    if all_c:
        new_c += all_c
        new_order.append(len(all_c))
    node.c = new_c

    # print('new order', new_order)
    # print('new c ', node.c)
    return new_order != old_order




