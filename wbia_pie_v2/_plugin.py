# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
from wbia.constants import ANNOTATION_TABLE, UNKNOWN
from wbia.constants import CONTAINERIZED, PRODUCTION  # NOQA
import numpy as np
import utool as ut
import wbia
import os
import torch
import torchvision.transforms as transforms  # noqa: E402
from scipy.spatial import distance_matrix

from wbia_pie_v2.default_config import get_default_config
from wbia_pie_v2.datasets import AnimalNameWbiaDataset  # noqa: E402
from wbia_pie_v2.metrics import eval_onevsall
from wbia_pie_v2.models import build_model
from wbia_pie_v2.utils import read_json
from wbia_pie_v2.metrics import pred_light

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']

DEMO_DB_URL = {
    'whale_shark': 'https://www.dropbox.com/s/xwac8lyyua5jw3o/whale_shark_cropped_demo.zip?dl=1'
}

# TODO: upload config to public server
CONFIGS = {
    'whale_shark': 'https://www.dropbox.com/s/6wvkrf319lhwhug/pie_v2.whale_shark.20210304.yaml?dl=1',
}

# TODO: upload models to public server
MODEL_URLS = {
    'whale_shark': None,
}


@register_ibs_method
def pie_embedding(ibs, aid_list, species=None, use_depc=True):
    r"""
    Generate embeddings using the Pose-Invariant Embedding (PIE)
    Args:
        ibs (IBEISController): IBEIS / WBIA controller object
        aid_list  (int): annot ids specifying the input
        species (string): name of species category.
                If None, use category of the first annotation. Default: None
        use_depc (bool): use dependency cache
    CommandLine:
        python -m wbia_pie_v2._plugin pie_embedding
    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import wbia_pie_v2
        >>> demo_db_url = 'https://www.dropbox.com/s/xwac8lyyua5jw3o/whale_shark_cropped_demo.zip?dl=1'
        >>> species = 'whale_shark'
        >>> subset = 'test2021'
        >>> test_ibs = wbia_pie_v2_test_ibs(demo_db_url, species, subset)
        >>> aid_list = test_ibs.get_valid_aids(species=species)
        >>> embs_depc    = np.array(ibs.pie_embedding(aids, use_depc=True))
        >>> embs_no_depc = np.array(ibs.pie_embedding(aids, use_depc=False))
        >>> diffs = np.abs(embs_depc - embs_no_depc)
        >>> assert diffs.max() < 1e-8
        >>> # each embedding is 512 floats long so we'll just check a bit
        >>> annot_uuids = ibs.get_annot_semantic_uuids(aids)
        >>> wanted_uuid = uuid.UUID('588dc49a-9b7f-d362-1667-1f9f002cd566')
        >>> wanted_index = annot_uuids.index(wanted_uuid)
        >>> assert wanted_index is not None and wanted_index in list(range(len(aids)))
        >>> result = embs_depc[wanted_index][:20]
        >>> result_ = np.array([-0.03839333,  0.01182338,  0.02393869, -0.07164327, -0.04367629,
        >>>                     -0.00150531,  0.01324393,  0.10909598,  0.02349781,  0.08439559,
        >>>                     -0.06415793,  0.0110384 ,  0.03897202, -0.11256221,  0.00709192,
        >>>                      0.10403764,  0.00615681, -0.10405623,  0.0320793 , -0.0394897 ])
        >>> assert result.shape == result_.shape
        >>> diffs = np.abs(result - result_)
        >>> assert diffs.max() < 1e-5
    Example:
        >>> # ENABLE_DOCTEST
        >>> # This tests that an aid's embedding is independent of the other aids processed in the same call
        >>> import wbia_pie_v2
        >>> ibs = wbia_pie_v2._plugin.pie_testdb_ibs()
        >>> aids = ibs.get_valid_aids(species='Mobula birostris')
        >>> aids1 = aids[:-1]
        >>> aids2 = aids[1:]
        >>> embs1 = ibs.pie_compute_embedding(aids1)
        >>> embs2 = ibs.pie_compute_embedding(aids2)
        >>> # just look at the overlapping aids/embs
        >>> embs1 = np.array(embs1[1:])
        >>> embs2 = np.array(embs2[:-1])
        >>> compare_embs = np.abs(embs1 - embs2)
        >>> assert compare_embs.max() < 1e-8
    """

    if use_depc:
        embeddings = ibs.depc_annot.get('PiePytorchEmbedding', aid_list, 'embedding')
    else:
        embeddings = pie_compute_embedding(ibs, aid_list)
    return embeddings


@register_preproc_annot(
    tablename='PiePytorchEmbedding',
    parents=[ANNOTATION_TABLE],
    colnames=['embedding'],
    coltypes=[np.ndarray],
    fname='pie_v2',
    chunksize=128,
)
@register_ibs_method
def pie_embedding_depc(depc, aid_list, config):
    ibs = depc.controller
    embs = pie_compute_embedding(ibs, aid_list)
    for aid, emb in zip(aid_list, embs):
        yield (np.array(emb),)


@register_ibs_method
def pie_compute_embedding(ibs, aid_list, species=None, use_gpu=False):
    # Get species from the first annotation if not specified
    if species is None:
        species = ibs.get_annot_species_texts(aid_list[0])

    # Load config
    cfg = _load_config(species, use_gpu)

    # Load model
    model = _load_model(cfg, MODEL_URLS[species])

    # Preprocess images to model input
    test_loader, test_dataset = _load_data(ibs, aid_list, cfg)

    # Compute embeddings
    embeddings = []
    model.eval()
    with torch.no_grad():
        for images, names in test_loader:
            if cfg.use_gpu:
                images = images.cuda(non_blocking=True)

            output = model(images.float())
            embeddings.append(output.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)
    return embeddings


def _load_config(species, use_gpu):
    r"""
    Load a configuration file for species
    """
    config_url = CONFIGS[species]
    config_fname = config_url.split('/')[-1]
    config_file = ut.grab_file_url(
        config_url, appname='wbia_pie_v2', check_hash=True, fname=config_fname
    )
    # TODO remove this after config is uploaded to public server
    config_file = 'wbia_pie_v2/reid-data/temp_data/pie_v2.whale_shark.20210304.yaml'

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    cfg.merge_from_file(config_file)
    return cfg


def _load_model(cfg, model_url=None):
    r"""
    Load a model based on config file
    """
    print('Building model: {}'.format(cfg.model.name))
    model = build_model(
        name=cfg.model.name,
        num_classes=cfg.model.num_train_classes,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
    )

    # Download the model weights
    if model_url is not None:
        model_fname = model_url.split('/')[-1]
        model_path = ut.grab_file_url(
            model_url, appname='wbia_pie_v2', check_hash=True, fname=model_fname
        )
    else:
        # TODO remove this after config is uploaded to public server
        model_path = 'wbia_pie_v2/reid-data/temp_data/model.pth.tar-200_cpu'

    if cfg.use_gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    print('Loaded model from {}'.format(model_path))
    if cfg.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    return model


def _load_data(ibs, aid_list, cfg):
    r"""
    Load data, preprocess and create data loaders
    """
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_paths = ibs.get_annot_image_paths(aid_list)
    bboxes = ibs.get_annot_bboxes(aid_list)
    names = ibs.get_annot_name_rowids(aid_list)
    target_imsize = (cfg.data.height, cfg.data.width)

    dataset = AnimalNameWbiaDataset(
        image_paths, names, bboxes, target_imsize, test_transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.data.workers,
        pin_memory=True,
        drop_last=False,
    )
    print('Loaded {} images for model evaluation'.format(len(dataset)))

    return dataloader, dataset


@register_ibs_method
def wbia_pie_v2_test_ibs(demo_db_url, species, subset):
    r"""
    Create a database to test orientation detection from a coco annotation file
    """
    # TODO upload test db to public server

    testdb_name = 'testdb_{}_{}'.format(species, subset)

    test_ibs = wbia.opendb(testdb_name, allow_newdir=True)
    if len(test_ibs.get_valid_aids()) > 0:
        return test_ibs
    else:
        # Download demo data archive
        db_dir = ut.grab_zipped_url(demo_db_url, appname='wbia_pie_v2')
        # Load coco annotations
        json_file = os.path.join(
            db_dir, 'annotations', 'instances_{}.json'.format(subset)
        )
        coco = read_json(json_file)
        coco_annots = coco['annotations']
        coco_images = coco['images']
        print('Found {} records in demo db'.format(len(coco_annots)))

        # Parse COCO annotations
        id2file = {a['id']: a['file_name'] for a in coco_images}
        files = [id2file[a['image_id']] for a in coco_annots]
        # Get image paths and add them to the database
        gpaths = [os.path.join(db_dir, 'images', subset, f) for f in files]
        names = [a['name'] for a in coco_annots]

        # Add files and names to db
        gid_list = test_ibs.add_images(gpaths)
        nid_list = test_ibs.add_names(names)
        species = [species] * len(gid_list)

        # these images are pre-cropped aka trivial annotations
        bbox_list = [a['bbox'] for a in coco_annots]
        test_ibs.add_annots(
            gid_list, bbox_list=bbox_list, species_list=species, nid_list=nid_list
        )

        return test_ibs


@register_ibs_method
def evaluate_1vsall_distmat(ibs, aid_list, species, use_gpu, ranks=[1, 5, 10, 20]):

    embs = np.array(pie_embedding(ibs, aid_list, species, use_gpu))
    print('Computing distance matrix ...')
    distmat = distance_matrix(embs, embs)

    print('Computing ranks ...')
    db_labels = np.array(ibs.get_annot_name_rowids(aid_list))
    cranks, mAP = eval_onevsall(distmat, db_labels, db_labels, use_sids=False)

    print('** Results **')
    # print('mAP: {:.1%}'.format(mAP))
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cranks[r - 1]))
    return cranks[0]


# The following functions are copied from PIE v1
# https://github.com/WildMeOrg/wbia-plugin-pie/wbia_pie/_plugin.py
# These functions are agnistic to the method of computing embeddings


def _db_labels_for_pie(ibs, daid_list):
    db_labels = ibs.get_annot_name_texts(daid_list)
    db_auuids = ibs.get_annot_semantic_uuids(daid_list)
    # later we must know which db_labels are for single auuids, hence prefix
    db_auuids = [UNKNOWN + str(auuid) for auuid in db_auuids]
    db_labels = [
        lab if lab is not UNKNOWN else auuid for lab, auuid in zip(db_labels, db_auuids)
    ]
    db_labels = np.array(db_labels)
    return db_labels


@register_ibs_method
def pie_predict_light(ibs, qaid, daid_list):
    db_embs = np.array(ibs.pie_embedding(daid_list))
    db_labels = np.array(ibs.get_annot_name_texts(daid_list))
    query_emb = np.array(ibs.pie_embedding([qaid]))

    ans = pred_light(query_emb, db_embs, db_labels)
    return ans


def _pie_accuracy(ibs, qaid, daid_list):
    daids = daid_list.copy()
    daids.remove(qaid)
    ans = ibs.pie_predict_light(qaid, daids)
    ans_names = [row['label'] for row in ans]
    ground_truth = ibs.get_annot_name_texts(qaid)
    try:
        rank = ans_names.index(ground_truth) + 1
    except ValueError:
        rank = -1
    print('rank %s' % rank)
    return rank


@register_ibs_method
def pie_mass_accuracy(ibs, aid_list, daid_list=None):
    if daid_list is None:
        daid_list = aid_list
    ranks = [_pie_accuracy(ibs, aid, daid_list) for aid in aid_list]
    return ranks


@register_ibs_method
def accuracy_at_k(ibs, ranks, max_rank=10):
    counts = [ranks.count(i) for i in range(1, max_rank + 1)]
    percent_counts = [count / len(ranks) for count in counts]
    cumulative_percent = [
        sum(percent_counts[:i]) for i in range(1, len(percent_counts) + 1)
    ]
    return cumulative_percent


@register_ibs_method
def subset_with_resights(ibs, aid_list, n=3):
    names = ibs.get_annot_name_rowids(aid_list)
    name_counts = _count_dict(names)
    good_annots = [aid for aid, name in zip(aid_list, names) if name_counts[name] >= n]
    return good_annots


def _count_dict(item_list):
    from collections import defaultdict

    count_dict = defaultdict(int)
    for item in item_list:
        count_dict[item] += 1
    return dict(count_dict)


def subset_with_resights_range(ibs, aid_list, min_sights=3, max_sights=10):
    name_to_aids = _name_dict(ibs, aid_list)
    final_aids = []
    import random

    for name, aids in name_to_aids.items():
        if len(aids) < min_sights:
            continue
        elif len(aids) <= max_sights:
            final_aids += aids
        else:
            final_aids += sorted(random.sample(aids, max_sights))
    return final_aids


@register_ibs_method
def pie_new_accuracy(ibs, aid_list, min_sights=3, max_sights=10):
    aids = subset_with_resights_range(ibs, aid_list, min_sights, max_sights)
    ranks = ibs.pie_mass_accuracy(aids)
    accuracy = ibs.accuracy_at_k(ranks)
    print(
        'Accuracy at k for annotations with %s-%s sightings:' % (min_sights, max_sights)
    )
    print(accuracy)
    return accuracy


def _name_dict(ibs, aid_list):
    names = ibs.get_annot_name_rowids(aid_list)
    from collections import defaultdict

    name_aids = defaultdict(list)
    for aid, name in zip(aid_list, names):
        name_aids[name].append(aid)
    return name_aids


def _invert_dict(d):
    from collections import defaultdict

    inverted = defaultdict(list)
    for key, value in d.items():
        inverted[value].append(key)
    return dict(inverted)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_orientation._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()