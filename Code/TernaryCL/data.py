import pathlib
import numpy as np
import json
import pdb

class load_data():
    def __init__(self, args):
        self.args = args
        self.data_files = self.args.data_files

        self.fetch_data()
        
    def get_phrases(self,file_path):
        f = open(file_path,"r").readlines()
        phrase2id = {}
        id2phrase = {}
        word2id = set()
        for line in f:
            phrase, ID = line.strip().split("\t")
            phrase2id[phrase] = int(ID)
            id2phrase[int(ID)] = phrase
            for word in phrase.split():
                word2id.add(word)
        return phrase2id,id2phrase,word2id

    def get_phrases_rel(self,file_path):
        f = open(file_path,"r").readlines()
        phrase2id = {}
        id2phrase = {}
        word2id = set()

        num_rel_ID = int(len(f))

        for line in f:
            phrase,ID = line.strip().split("\t")
            phrase2id[phrase] = int(ID)
            id2phrase[int(ID)] = phrase
            for word in phrase.split():
                word2id.add(word)

            # pdb.set_trace()
            r_inv = "inversed " + phrase
            if r_inv not in phrase2id:
                phrase2id[r_inv] = num_rel_ID
                id2phrase[num_rel_ID] = r_inv
                num_rel_ID += 1

        # pdb.set_trace()
        # add self relation
        r_inv = "self"
        phrase2id[r_inv] = num_rel_ID
        id2phrase[num_rel_ID] = r_inv
        num_rel_ID += 1

        return phrase2id,id2phrase,word2id

    
    def get_phrase2word(self,phrase2id,word2id):
        phrase2word = {}
        for phrase in phrase2id:
            words = []
            for word in phrase.split(): words.append(word2id[word])
            phrase2word[phrase2id[phrase]] = words
        return phrase2word

    
    def get_word_embed(self,word2id,GLOVE_PATH):
        word_embed = {}
        if pathlib.Path(GLOVE_PATH).is_file():
            print("utilizing pre-trained word embeddings")
            with open(GLOVE_PATH, encoding="utf8") as f:
                for line in f:
                    word, vec = line.split(' ', 1)
                    if word in word2id:
                        word_embed[word2id[word]] = np.fromstring(vec, sep=' ')
        else: print("word embeddings are randomly initialized")

        wordkeys = word2id.values()
        a = [words for words in wordkeys if words not in word_embed]
        for word in a:
            word_embed[word] = np.random.normal(size = 300)

        self.embed_matrix  = np.zeros((len(word_embed),300))
        for word in word_embed:
            self.embed_matrix[word] = word_embed[word]

    def get_self_triples(self, id2ent, rel2id):
        # pdb.set_trace()
        self_rel_id = rel2id["self"]
        triples = []
        for ent_id in id2ent:
            trip = str(ent_id) + "\t" + str(self_rel_id) + "\t" + str(ent_id)
            triples.append(trip)

        return triples


    def get_train_triples(self,triples_path,entid2clustid,rel2id,id2rel, self_triples):
        trip_list = []
        label_graph = {}
        label_graph_rel = {}
        neighbors_incoming = {}
        neighbors_outgoing = {}
        self.label_filter = {}
        f = open(triples_path,"r").readlines()
        for trip in f:
            trip = trip.strip().split()
            e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
            r_inv = "inversed " + id2rel[r]
            r_inv = rel2id[r_inv]
                
            if (e1,r) not in label_graph:
                label_graph[(e1,r)] = set()
            label_graph[(e1,r)].add(e2)

            if (e1,e2) not in label_graph_rel:
                label_graph_rel[(e1,e2)] = set()
            label_graph_rel[(e1,e2)].add(r)

            # incoming edges
            if e1 not in neighbors_outgoing:
                neighbors_outgoing[e1] = set()
            neighbors_outgoing[e1].add((e2,r))

            if e2 not in neighbors_incoming:
                neighbors_incoming[e2] = set()
            neighbors_incoming[e2].add((e1,r))

            if (e2,r_inv) not in label_graph:
                label_graph[(e2,r_inv)] = set()
            label_graph[(e2,r_inv)].add(e1)

            if (e2,e1) not in label_graph_rel:
                label_graph_rel[(e2,e1)] = set()
            label_graph_rel[(e2,e1)].add(r_inv)

            if e2 not in neighbors_outgoing:
                neighbors_outgoing[e2] = set()
            neighbors_outgoing[e2].add((e1, r_inv))

            if e1 not in neighbors_incoming:
                neighbors_incoming[e1] = set()
            neighbors_incoming[e1].add((e2, r_inv))

            
            if (e1,r) not in self.label_filter:
                self.label_filter[(e1,r)] = set()
            self.label_filter[(e1,r)].add(entid2clustid[e2])
            
            if (e2,r_inv) not in self.label_filter:
                self.label_filter[(e2,r_inv)] = set()
            self.label_filter[(e2,r_inv)].add(entid2clustid[e1])

            trip_list.append([e1,r,e2])
            trip_list.append([e2,r_inv,e1])

        for trip in self_triples:
            trip = trip.strip().split()
            e1, r, e2 = int(trip[0]), int(trip[1]), int(trip[2])

            if (e1, r) not in label_graph:
                label_graph[(e1, r)] = set()
            label_graph[(e1, r)].add(e2)

            if (e1,e2) not in label_graph_rel:
                label_graph_rel[(e1,e2)] = set()
            label_graph_rel[(e1,e2)].add(r)

            # incoming edges
            if e1 not in neighbors_outgoing:
                neighbors_outgoing[e1] = set()
                neighbors_incoming[e1] = set()
            neighbors_outgoing[e1].add((e2, r))
            neighbors_incoming[e1].add((e2, r))

            if (e1, r) not in self.label_filter:
                self.label_filter[(e1, r)] = set()
            self.label_filter[(e1, r)].add(entid2clustid[e2])

            trip_list.append([e1, r, e2])

        return np.array(trip_list), label_graph, label_graph_rel, neighbors_incoming, neighbors_outgoing
    
    def get_test_triples(self, triples_path,rel2id,id2rel):
        trip_list = []
        f = open(triples_path,"r").readlines()
        for trip in f:
            trip = trip.strip().split()
            e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
            r_inv = "inversed " + id2rel[r]
            trip_list.append([e1,r,e2])
            trip_list.append([e2,rel2id[r_inv],e1])
        return np.array(trip_list)
    
    def get_clusters(self,clust_path):
        ent_clusts = {}
        entid2clustid = {}
        ent_list = []
        unique_clusts = []
        ID = 0
        f = open(clust_path,"r").readlines()
        for line in f:
            line = line.strip().split()
            clust = [int(ent) for ent in line[2:]]
            ent_clusts[int(line[0])] = clust
            if line[0] not in ent_list:
                unique_clusts.append(clust)
                ent_list.extend(line[2:])
                for ent in clust: entid2clustid[ent] = ID
                ID += 1

        print("cluster num:" + str(ID))
        return ent_clusts,entid2clustid,unique_clusts

    def get_kgid2synkgid(self, file_path, ent2id, rel2id):
        # pdb.set_trace()
        with open(file_path, 'r') as load_f:
            temp = json.load(load_f)
            syn_rel_id = rel2id["syn"]
            SYN_triples = []
            for key in temp:
                # pdb.set_trace()
                key_id = ent2id[key]
                for okgtext in temp[key]:
                    okgtext_id = ent2id[okgtext]
                    trip = str(key_id) + "\t" + str(syn_rel_id) + "\t" + str(okgtext_id)
                    SYN_triples.append(trip)
        return SYN_triples
    
    def fetch_data(self):
        self.rel2id,self.id2rel,self.word2id = self.get_phrases_rel(self.data_files["rel2id_path"])
        self.ent2id,self.id2ent,self.ent_word2id = self.get_phrases(self.data_files["ent2id_path"])

        self.true_clusts, self.entid2clustid,_ = self.get_clusters(self.data_files["gold_npclust_path"])


        self_triples = self.get_self_triples(self.id2ent, self.rel2id)
        SYN_triples = self.get_kgid2synkgid(self.data_files["sd_kgtext2okgtext_ent"], self.ent2id, self.rel2id)
        self_triples.extend(SYN_triples)
        self.train_trips,self.label_graph, self.label_graph_rel,\
        self.neighbors_incoming, self.neighbors_outgoing = self.get_train_triples(self.data_files["train_trip_path"],
                                                                               self.entid2clustid,self.rel2id, 
                                                                               self.id2rel, self_triples)

        self.test_trips = self.get_test_triples(self.data_files["test_trip_path"],self.rel2id, self.id2rel)
        self.valid_trips = self.get_test_triples(self.data_files["valid_trip_path"],self.rel2id, self.id2rel)
        

        self.word2id = self.word2id.union(self.ent_word2id)
        self.word2id.add("inversed")
        self.word2id = {word: index for index, word in enumerate(list(self.word2id))}
        self.word2id['<PAD>'] = len(self.word2id)
        self.rel2word = self.get_phrase2word(self.rel2id, self.word2id)
        self.ent2word = self.get_phrase2word(self.ent2id, self.word2id)

        self.get_word_embed(self.word2id, self.data_files["glove_path"])


