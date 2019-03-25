from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data

import numpy as np
import tensorflow as tf
from models.toxicity_clasifier_keras import ToxicityClassifierKeras
from agents.smart_replace import smart_replace , get_possible_replace
from attacks.hot_flip import HotFlip
import time

def check_num_of_flips(best_flip_status):
    pointer = best_flip_status
    count = 0
    while pointer != None:
        count += 1
        pointer = pointer.prev_flip_status

    return count - 1

def main():

    # get restore model
    sess = tf.Session()
    tox_model = ToxicityClassifierKeras(session=sess)

    hot_flip = HotFlip(model=tox_model,break_on_half = True,beam_search_size = 10)
    hot_flip_tox = HotFlip(model=tox_model,use_tox_as_score = True,break_on_half = True , calc_tox_for_beam = True
                           , beam_search_size = 3)

    #hot_dup = HotFlip(model=tox_model,attack_mode='dup')

    list_of_attack = [ hot_flip , hot_flip_tox ]

    # get data
    dataset = data.Dataset.init_from_dump()

    index_of_toxic_sent = np.where(dataset.train_lbl[:, 0] == 1)[0]

    list_flip_attack = []
    list_dup_attack = []

    list_final_tox_flip = []
    list_final_tox_dup = []

    list_flip_attack_time = []
    list_dup_attack_time = []

    #check_length(dataset)
    num_of_sentence_to_attack = 100
    for i in range (num_of_sentence_to_attack):

        index_to_attack = index_of_toxic_sent[i]

        # taking the first sentence.
        seq = np.expand_dims(dataset.train_seq[index_to_attack, :], 0)


        for attack in list_of_attack:
            print("attack use_tox_as_score mode: " , attack.use_tox_as_score)
            #do hot flip attack
            t = time.time()
            best_flip_status , _ = attack.attack(seq = seq)
            dur = time.time() - t

            # print sentance after the flips
            # print("flipped sentence: ")
            # print(data.seq_2_sent(best_flip_status.fliped_sent, char_to_token_dic))

            # classes before the change
            # print("tox class before the flip: ")
            # classes = tox_model.classify(seq)[0][0]
            # print(classes)

            # classes after the change
            print("tox class after the flip: ")
            classes = tox_model.classify(np.expand_dims(best_flip_status.fliped_sent, 0))[0][0]
            print(classes)

            num_flips = check_num_of_flips(best_flip_status)

            if attack.use_tox_as_score == False:
                list_flip_attack.append( num_flips  )
                list_final_tox_flip.append( classes  )
                list_flip_attack_time.append(dur)
            else:
                list_dup_attack.append( num_flips )
                list_final_tox_dup.append( classes  )
                list_dup_attack_time.append(dur)


        print("list_flip_attack")
        print(list_flip_attack)
        print("mean num of flips: " , sum(list_flip_attack) / float(len(list_flip_attack)))
        print("mean final class: ", sum(list_final_tox_flip) / float(len(list_final_tox_flip)))
        print("mean run time: ", sum(list_flip_attack_time) / float(len(list_flip_attack_time)))

        print("list_dup_attack")
        print(list_dup_attack)
        print("mean num of dups: ", sum(list_dup_attack) / float(len(list_dup_attack)))
        print("mean final class: ", sum(list_final_tox_dup) / float(len(list_final_tox_dup)))
        print("mean run time: ", sum(list_dup_attack_time) / float(len(list_dup_attack_time)))


if __name__ == '__main__':
    main()
