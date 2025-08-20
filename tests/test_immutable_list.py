import pytest

from xdsl.utils.immutable_list import IList


def test_append():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j])
    list.append(k)
    assert list == IList([i, j, k])


def test_append_to_frozen():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j])
    list.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        list.append(k)


def test_extend():
    i, j, k = 1, 2, 3
    list0: IList[int] = IList([i, j, k])
    list1: IList[int] = IList([i, j, k])
    list0.extend(list1)
    assert list0 == IList([i, j, k, i, j, k])


def test_extend_frozen():
    i, j, k = 1, 2, 3
    list0: IList[int] = IList([i, j, k])
    list1: IList[int] = IList([i, j, k])
    list0.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        list0.extend(list1)


def test_insert():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j])
    list.insert(1, k)
    assert list == IList([i, k, j])


def test_insert_frozen():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j])
    list.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        list.insert(1, k)


def test_remove():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    list.remove(k)
    assert list == IList([i, j])


def test_remove_frozen():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    list.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        list.remove(k)


def test_pop():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    assert list.pop(-1) == k
    assert list == IList([i, j])


def test_pop_frozen():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    list.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        list.pop(-1)


def test_clear():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    list.clear()
    assert list == IList(())


def test_clear_frozen():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    list.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        list.clear()


def test_setitem():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    list[1] = 4
    assert list == IList([i, 4, k])


def test_setitem_frozen():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    list.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        list[1] = 4


def test_delitem():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    del list[1]
    assert list == IList([i, k])


def test_delitem_frozen():
    i, j, k = 1, 2, 3
    list: IList[int] = IList([i, j, k])
    list.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        del list[1]


def test_add():
    i, j, k = 1, 2, 3
    list0: IList[int] = IList([i, j])
    list1: IList[int] = IList([k])
    list2 = list0 + list1
    assert list2 == IList([i, j, k])


def test_add_frozen():
    i, j, k = 1, 2, 3
    list0: IList[int] = IList([i, j])
    list1: IList[int] = IList([k])
    list0.freeze()
    list1.freeze()
    list2 = list0 + list1
    assert list2 == IList([i, j, k])


def test_iadd():
    i, j, k = 1, 2, 3
    list0: IList[int] = IList([i, j])
    list1: IList[int] = IList([k])
    list0 += list1
    assert list0 == IList([i, j, k])


def test_iadd_frozen():
    i, j, k = 1, 2, 3
    list0: IList[int] = IList([i, j])
    list1: IList[int] = IList([k])
    list0.freeze()
    with pytest.raises(Exception, match="frozen list can not be modified"):
        list0 += list1


def test_eq():
    i, j, k = 1, 2, 3
    list0: IList[int] = IList([i, j, k])
    list1: IList[int] = IList([i, j, k])
    assert list0 == list1
