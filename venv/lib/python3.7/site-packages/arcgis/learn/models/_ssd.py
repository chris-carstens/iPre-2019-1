try:
    import torch
    from fastai.vision.learner import create_cnn
    from fastai.callbacks.hooks import model_sizes
    from fastai.vision.learner import create_body
    from torchvision.models import resnet34
    import numpy as np
    from ._ssd_utils import SSDHead, BCE_Loss, FocalLoss, one_hot_embedding, nms
    from .._data import prepare_data
    import json
    import os
    from pathlib import Path
    from ._codetemplate import code
    HAS_FASTAI = True
except Exception as e:
    HAS_FASTAI = False

def _raise_fastai_import_error():
    raise Exception('This module requires fastai, PyTorch and torchvision as its dependencies. Install it using "conda install -c pytorch -c fastai fastai pytorch torchvision"')

_EMD_TEMPLATE = {
    "Framework": "arcgis.learn.models._inferencing",
    "InferenceFunction":"ArcGISObjectDetector.py",
    "ModelConfiguration": "_DynamicSSD",
    "ModelFile":"",
    "ModelType":"ObjectDetection",
    "ImageHeight":None,
    "ImageWidth":None,
    "ExtractBands":[0,1,2],
    "Grids":None,
    "Zooms":None,
    "Ratios":None,
    "Classes" : []
}

_CLASS_TEMPLATE = {
        "Value": 0,
        "Name": "Pool",
        "Color": [0, 255, 0]
      }

class _EmptyData():
    def __init__(self, path, c, loss_func, chip_size):
        self.path = path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.c = c
        self.loss_func = loss_func
        self.chip_size = chip_size

class SingleShotDetector(object):

    """
    Creates a Single Shot Detector with the specified grid sizes, zoom scales
    and aspect  ratios. Based on Fast.ai MOOC Version2 Lesson 9.

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            `prepare_data` function.
    ---------------------   -------------------------------------------
    grids                   Required list. Grid sizes used for creating anchor
                            boxes.
    ---------------------   -------------------------------------------
    zooms                   Optional list. Zooms of anchor boxes.
    ---------------------   -------------------------------------------
    ratios                  Optional list of tuples. Aspect ratios of anchor
                            boxes.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used for
                            creating the base of the `SingleShotDetector`, which
                            is `resnet34` by default.
    ---------------------   -------------------------------------------
    dropout                 Optional float. Dropout propbability. Increase it to
                            reduce overfitting.
    ---------------------   -------------------------------------------
    bias                    Optional float. Bias for SSD head.
    ---------------------   -------------------------------------------
    focal_loss              Optional boolean. Uses Focal Loss if True.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    :returns: `SingleShotDetector` Object
    """

    def __init__(self, data, grids=[4, 2, 1], zooms=[0.7, 1., 1.3], ratios=[[1., 1.], [1., 0.5], [0.5, 1.]],
                 backbone=None, drop=0.3, bias=-4., focal_loss=False, pretrained_path=None):

        super().__init__()

        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if not HAS_FASTAI:
            _raise_fastai_import_error()

        if backbone is None:
            backbone = resnet34

        self._create_anchors(grids, zooms, ratios)

        num_features = model_sizes(create_body(backbone), size=(data.chip_size, data.chip_size))[-1][-1]

        ssd_head = SSDHead(grids, self._anchors_per_cell, data.c, num_features=num_features, drop=drop, bias=bias)

        self._data = data
        self.learn = create_cnn(data=data, arch=backbone, custom_head=ssd_head)
        self.learn.model = self.learn.model.to(self._device)

        if pretrained_path is not None:
            self.load(pretrained_path)

        if focal_loss:
            self._loss_f = FocalLoss(data.c)
        else:
            self._loss_f = BCE_Loss(data.c)

        self.learn.loss_func = self._ssd_loss

    @classmethod
    def from_emd(cls, data, emd_path):
        """
        Creates a Single Shot Detector from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from `prepare_data` function or None for
                                inferencing.
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Esri Model Definition
                                file.
        =====================   ===========================================

        :returns: `SingleShotDetector` Object
        """
        emd_path = Path(emd_path)
        emd = json.load(open(emd_path))
        model_file = Path(emd['ModelFile'])
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        class_mapping = {i['Value'] : i['Name'] for i in emd['Classes']}
        if data is None:
            empty_data = _EmptyData(path='str', loss_func=None, c=len(class_mapping) + 1, chip_size=emd['ImageHeight'])
            return cls(empty_data, emd['Grids'], emd['Zooms'], emd['Ratios'], pretrained_path=str(model_file))
        else:
            return cls(data, emd['Grids'], emd['Zooms'], emd['Ratios'], pretrained_path=str(model_file))


    def lr_find(self):
        """
        Runs the Learning Rate Finder, and displays the graph of it's output.
        Helps in choosing the optimum learning rate for training the model.
        """
        from IPython.display import clear_output
        self.learn.lr_find()
        clear_output()
        self.learn.recorder.plot()

    def fit(self, epochs=10, lr=slice(1e-4,3e-3)):
        """
        Train the model for the specified number of epocs and using the
        specified learning rates

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        epochs                  Required integer. Number of cycles of training
                                on the data. Increase it if underfitting.
        ---------------------   -------------------------------------------
        lr                      Required float or slice of floats. Learning rate
                                to be used for training the model. Select from
                                the `lr_find` plot.
        =====================   ===========================================
        """
        self.learn.fit(epochs, lr)

    def unfreeze(self):
        """
        Unfreezes the earlier layers of the detector for fine-tuning.
        """
        self.learn.unfreeze()

    def _create_anchors(self, anc_grids, anc_zooms, anc_ratios):

        self.grids = anc_grids
        self.zooms = anc_zooms
        self.ratios =  anc_ratios

        anchor_scales = [(anz*i, anz*j) for anz in anc_zooms for (i,j) in anc_ratios]

        self._anchors_per_cell = len(anchor_scales)

        anc_offsets = [1/(o*2) for o in anc_grids]

        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), self._anchors_per_cell, axis=0)

        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                       for ag in anc_grids])

        self._grid_sizes = torch.Tensor(np.concatenate([np.array([ 1/ag  for i in range(ag*ag) for o,p in anchor_scales])
                       for ag in anc_grids])).unsqueeze(1).to(self._device)

        self._anchors = torch.Tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self._device)

        self._anchor_cnr = self._hw2corners(self._anchors[:,:2], self._anchors[:,2:])

    def _hw2corners(self, ctr, hw):
        return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)

    def _get_y(self, bbox, clas):
        bbox = bbox.view(-1,4) #/sz
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def _actn_to_bb(self, actn, anchors, grid_sizes):
        actn_bbs = torch.tanh(actn)
        actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
        return self._hw2corners(actn_centers, actn_hw)

    def _map_to_ground_truth(self, overlaps, print_it=False):
        prior_overlap, prior_idx = overlaps.max(1)
        if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap, gt_idx


    def _ssd_1_loss(self, b_c, b_bb, bbox, clas, print_it=False):
        bbox,clas = self._get_y(bbox,clas)
        bbox = self._normalize_bbox(bbox)

        a_ic = self._actn_to_bb(b_bb, self._anchors, self._grid_sizes)
        overlaps = self._jaccard(bbox.data, self._anchor_cnr.data)
        try:
            gt_overlap,gt_idx = self._map_to_ground_truth(overlaps,print_it)
        except Exception as e:
            return 0.,0.
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = 0
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self._loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def _ssd_loss(self, pred, targ1, targ2, print_it=False):
        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred, targ1, targ2):
            loc_loss,clas_loss = self._ssd_1_loss(b_c,b_bb,bbox.to(self._device),clas.to(self._device),print_it)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print('loc: {lls}, clas: {lcs}'.format(lls=lls, lcs=lcs))
        return lls+lcs

    def _intersect(self,box_a, box_b):
        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def _box_sz(self, b):
        return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def _jaccard(self, box_a, box_b):
        inter = self._intersect(box_a, box_b)
        union = self._box_sz(box_a).unsqueeze(1) + self._box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def _normalize_bbox(self, bbox):
        return (bbox+1.)/2.

    def _create_zip(self, zipname, path):
        import shutil
        zip_file = shutil.make_archive(zipname, 'zip', path)
        shutil.move(zip_file, path)

    def _create_emd(self, path):
        import random
        _EMD_TEMPLATE['ModelFile'] = path.name
        _EMD_TEMPLATE['ImageHeight'] = self._data.chip_size
        _EMD_TEMPLATE['ImageWidth'] = self._data.chip_size
        _EMD_TEMPLATE['Grids'] = self.grids
        _EMD_TEMPLATE['Zooms'] = self.zooms
        _EMD_TEMPLATE['Ratios'] = self.ratios
        for i, class_name in enumerate(self._data.classes[1:]): # 0th index is background
            _CLASS_TEMPLATE["Value"] = i
            _CLASS_TEMPLATE["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)]
            _CLASS_TEMPLATE["Color"] = color
            _EMD_TEMPLATE['Classes'].append(_CLASS_TEMPLATE.copy())

        json.dump(_EMD_TEMPLATE, open(path.with_suffix('.emd'), 'w'), indent=4)
        return path.stem

    def save(self, name_or_path):
        """
        Saves the model weights, creates an Esri Model Definition and Deep
        Learning Package zip for deployment to Image Server or ArcGIS Pro

        Train the model for the specified number of epocs and using the
        specified learning rates

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Name of the model to save. It
                                stores it at the pre-defined location. If path
                                is passed then it stores at the specified path
                                with model name as directory name. and creates
                                all the intermediate directories.
        =====================   ===========================================
        """
        if '\\' in name_or_path or '/' in name_or_path:
            path = Path(name_or_path)
            name = path.parts[-1]
            # to make fastai save to both path and with name
            temp = self.learn.path
            self.learn.path = path
            self.learn.model_dir = ''
            if not os.path.exists(self.learn.path):
                os.makedirs(self.learn.path)
            saved_path = self.learn.save(name, return_path=True)
            # undoing changes to self.learn.path and self.learn.model
            self.learn.path = temp
            self.learn.model_dir = 'models'
        else:
            temp = self.learn.path
            # fixing fastai bug
            self.learn.path = self.learn.path.parent
            self.learn.model_dir =  Path(self.learn.model_dir) /  name_or_path
            if not os.path.exists(self.learn.path / self.learn.model_dir):
                os.makedirs(self.learn.path / self.learn.model_dir)
            saved_path = self.learn.save(name_or_path,  return_path=True)
            # undoing changes to self.learn.path
            self.learn.path = temp
            self.learn.model_dir = 'models'

        zip_name = self._create_emd(saved_path)
        with open(saved_path.parent / _EMD_TEMPLATE['InferenceFunction'], 'w') as f:
            f.write(code)
        self._create_zip(zip_name, str(saved_path.parent))
        print('Created model files at {spp}'.format(spp=saved_path.parent))


    def load(self, name_or_path):
        """
        Loads a saved model for inferencing or fine tuning from the specified
        path or model name.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Name of the model to load from
                                the pre-defined location. If path is passed then
                                it loads from the specified path with model name
                                as directory name. Path to ".pth" file can also
                                be passed
        =====================   ===========================================
        """
        if '\\' in name_or_path or '/' in name_or_path:
            path = Path(name_or_path)
            # to make fastai from both path and with name
            temp = self.learn.path
            if path.is_file():
                name = path.stem
                self.learn.path = path.parent
            else:
                name = path.parts[-1]
                self.learn.path = path
            self.learn.model_dir = ''
            self.learn.load(name)
            # undoing changes to self.learn.path and self.learn.model_dir
            self.learn.path = temp
            self.learn.model_dir = 'models'
        else:
            temp = self.learn.path
            # fixing fastai bug
            self.learn.path = self.learn.path.parent
            self.learn.model_dir =  Path(self.learn.model_dir) /  name_or_path
            self.learn.load(name_or_path)
            # undoing changes to self.learn.path
            self.learn.path = temp
            self.learn.model_dir = 'models'

    def show_results(self, rows=5, thresh=0.5, nms_overlap=0.1):
        """
        Displays the results of a trained model on a part of the validation set.
        """
        self.learn.show_results(rows=rows, thresh=thresh, nms_overlap=nms_overlap, ssd=self)



