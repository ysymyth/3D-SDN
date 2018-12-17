import os

worldIds = ['0001', '0002', '0006', '0018', '0020']
sceneIds = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone', 'fog',
            'morning', 'overcast', 'rain', 'sunset']
worldSizes = [446, 232, 269, 338, 836]  # 0-446, including 446
category = ['Misc', 'Building', 'Car', 'GuardRail', 'Pole', 'Road', 'Sky', 'Terrain',
            'TrafficLight', 'TrafficSign', 'Tree', 'Truck', 'Van', 'Vegetation']


def get_tables(opt, datadir):
    """
    Get the mapping from (worldId, sceneId, rgb) to the semantic/instance ID.
    The instance ID is uniquely assigned to each car and van in the dataset.
    :param opt: 'segm' or 'inst'
    :param datadir: the dataset root
    :return:
    """
    global_obj_id = 0
    table_inst = {}
    table_segm = {}
    for worldId in worldIds:
        for sceneId in sceneIds:
            with open(os.path.join(datadir, "vkitti_1.3.1_scenegt",
                                   "%s_%s_scenegt_rgb_encoding.txt" % (worldId, sceneId)), 'r') as fin:
                first_line = True
                for line in fin:
                    if first_line:
                        first_line = False
                    else:
                        name, r, g, b = line.split(' ')
                        r, g, b = int(r), int(g), int(b)
                        if name.find(':') == -1:
                            table_segm[(worldId, sceneId, r, g, b)] = category.index(name)
                            table_inst[(worldId, sceneId, r, g, b)] = category.index(name)
                        else:
                            global_obj_id += 1
                            table_segm[(worldId, sceneId, r, g, b)] = category.index(name.split(':')[0])
                            table_inst[(worldId, sceneId, r, g, b)] = 5000 * category.index(
                                name.split(':')[0]) + global_obj_id

    return table_segm if opt == 'segm' else table_inst


def get_lists(opt):
    """
    Get the training/testing split for Virtual KITTI.
    :param opt: 'train' or 'test'
    :return:
    """
    splitRanges = {'train': [range(0, 356),     range(0, 185),      range(69, 270),     range(0, 270),      range(167, 837)],
                   'test':  [range(356, 447),   range(185, 233),    range(0, 69),       range(270, 339),    range(0, 167)],
                   'all':   [range(0, 447),     range(0, 233),      range(0, 270),     range(0, 339),      range(0, 837)]}
    _list = []
    for worldId in worldIds:
        for sceneId in sceneIds:
            for imgId in splitRanges[opt][worldIds.index(worldId)]:
                _list += ['%s/%s/%05d.png' % (worldId, sceneId, imgId)]
    return _list

