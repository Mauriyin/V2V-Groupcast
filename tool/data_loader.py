import pandas as pd


def load_data(leader_csv, member_csv):
    rb_leader = pd.read_csv(leader_csv)
    rb_member = pd.read_csv(member_csv)
    len_train = int(len(rb_leader))
    rb_leader_train = rb_leader.values.reshape(
        rb_leader.shape[0], rb_leader.shape[1])[:len_train, :]
    rb_member_train = rb_member.values.reshape(
        rb_member.shape[0], rb_member.shape[1])[1:len_train, :]

    return rb_leader_train, rb_member_train


def load_data_split(leader_csv, member_csv):
    rb_leader = pd.read_csv(leader_csv)
    rb_member = pd.read_csv(member_csv)
    len_train = int(len(rb_leader) / 10 * 7)
    rb_leader_train = rb_leader.values.reshape(
        rb_leader.shape[0], rb_leader.shape[1])[:len_train, :]
    rb_member_train = rb_member.values.reshape(
        rb_member.shape[0], rb_member.shape[1])[1:len_train, :]

    rb_leader_test = rb_leader.values.reshape(
        rb_leader.shape[0], rb_leader.shape[1])[len_train + 1:, :]
    rb_member_test = rb_member.values.reshape(
        rb_member.shape[0], rb_member.shape[1])[len_train + 2:, :]

    return rb_leader_train, rb_member_train, rb_leader_test, rb_member_test