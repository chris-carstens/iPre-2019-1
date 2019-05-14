'''
The orthomapping python API allows automating orthomapping tasks in the server environment.

For more information about orthomapping workflows in ArcGIS, please visit the help documentation at 
http://desktop.arcgis.com/en/arcmap/10.4/manage-data/raster-and-images/block-adjustment-for-mosaic-datasets.htm.
'''


import arcgis
import json
import string as _string
import random as _random
from arcgis.gis import Item
import collections
from ._util import _set_context

from arcgis.geoprocessing._support import _analysis_job, _analysis_job_results, \
                                          _analysis_job_status, _layer_input

###################################################################################################
###
### INTERNAL FUNCTIONS
###
###################################################################################################

def _execute_task(gis, taskname, params):

    gptool_url = gis.properties.helperServices.orthoMapping.url
    gptool = arcgis.gis._GISResource(gptool_url, gis)
    task = taskname

    task_url, job_info, job_id = _analysis_job(gptool, task, params)
    #print ('task url is ', task_url)

    job_info = _analysis_job_status(gptool, task_url, job_info)
    job_values = _analysis_job_results(gptool, task_url, job_info, job_id)

    item_properties = {
        "properties":{
            "jobUrl": task_url + '/jobs/' + job_info['jobId'],
            "jobType": "GPServer",
            "jobId": job_info['jobId'],
            "jobStatus": "completed"
            }
        }
    return job_values


#def _id_generator(size=6, chars=_string.ascii_uppercase + _string.digits):
    #return ''.join(_random.choice(chars) for _ in range(size))
###################################################################################################
###################################################################################################
def _set_image_collection_param(gis, params, image_collection):
    if isinstance(image_collection, str):
        if 'http:' in image_collection or 'https:' in image_collection:
            params['imageCollection'] = json.dumps({ 'url' : image_collection })
        else:
            params['imageCollection'] = json.dumps({ 'uri' : image_collection })
    elif isinstance(image_collection, Item):
        params['imageCollection'] = json.dumps({ "itemId" : image_collection.itemid })
    else:
        raise TypeError("image_collection should be a string (service name) or Item")

    return


def _create_output_image_service(gis, output_name, task):
    ok = gis.content.is_service_name_available(output_name, "Image Service")
    if not ok:
        raise RuntimeError("An Image Service by this name already exists: " + output_name)

    create_parameters = {
        "name": output_name,
        "description": "",
        "capabilities": "Image",
        "properties": {
            "path": "@",
            "description": "",
            "copyright": ""
        }
    }

    output_service = gis.content.create_service(output_name, create_params=create_parameters,
                                                      service_type="imageService")
    description = "Image Service generated from running the " + task + " tool."
    item_properties = {
        "description": description,
        "tags": "Analysis Result, " + task,
        "snippet": "Analysis Image Service generated from " + task
    }
    output_service.update(item_properties)
    return output_service

###################################################################################################
###
### PUBLIC API 
###
###################################################################################################
def is_supported(gis=None):
    """
    Returns True if the GIS supports orthomapping. If a gis isn't specified,
    checks if arcgis.env.active_gis supports raster analytics
    """
    gis = arcgis.env.active_gis if gis is None else gis
    if 'orthoMapping' in gis.properties.helperServices:
        return True
    else:
        return False

###################################################################################################
## Compute Sensor model
###################################################################################################
def compute_sensor_model(image_collection, 
                         mode='QUICK', 
                         location_accuracy='High', 
                         context=None,
                         *,
                         gis=None,
                         **kwargs):
    """
    compute_sensor_model computes the bundle block adjustment for the image collection 
    and applies the frame xform to the images. It will also generate the control point 
    table, solution table, solution points table and flight path table. 
    These tables will not be published as Portal items. 

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, the input image collection on which to compute
                           the sensor model.
                           The image_collection can be a portal Item or an image service URL or a URI
                           
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    mode                   Optional string.  the mode to be used for bundle block adjustment
                           Only the following modes are supported:
                           
                           - 'Quick' : Computes tie points and adjustment at 8x of the source imagery resolution

                           - 'Full'  : adjust the images in Quick mode then at 1x of the source imagery resolution

                           - 'Refine' : adjust the image at 1x of the source imagery resolution

                           By default, 'Quick' mode is applied to compute the sensor model.
    ------------------     --------------------------------------------------------------------
    location_acuracy       Optional string. this option allows users to specify the GPS location accuracy level of the  
                           source image. It determines how far the underline tool will search for neighboring 
                           matching images, then calculate tie points and compute adjustments.

                           Possible values for location_accuracy are:

                           - 'High'    : GPS accuracy is 0 to 10 meters, and the tool uses a maximum of 4 by 3 images 

                           - 'Medium'  : GPS accuracy of 10 to 20 meters, and the tool uses a maximum of 4 by 6 images

                           - 'Low'     : GPS accuracy of 20 to 50 meters, and the tool uses a maximum of 4 by 12 images 

                           - 'Verylow' : GPS accuracy is more than 50 meters, and the tool uses a maximum of 4 by 20 images

                           The default location_accuracy is 'High' 
    ------------------     --------------------------------------------------------------------
    context                Optional dictionary. The context parameter is used to configure additional client settings 
                           for block adjustment. The supported configurable parameters are for compute mosaic dataset
                           candidates after the adjustment. 

                           Example: 
                           {
                           "computeCandidate": False,
                           "maxoverlap": 0.6,
                           "maxloss": 0.05,
                           }
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================

    :return:
        The imagery layer url

    """
    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    _set_image_collection_param(gis, params, image_collection)
    modeAllowedValues = ["FULL","QUICK","REFINE"]
    if not mode.upper() in modeAllowedValues:
        raise RuntimeError("mode can only be one of the following: Quick, Full or Refine")

    params['mode'] = mode

    locationAccAllowedValues = ['HIGH', 'MEDIUM', 'LOW', 'VERYLOW']
    if not location_accuracy.upper() in locationAccAllowedValues:
        raise RuntimeError('location_accuracy can only be one of the following: '+ str(locationAccAllowedValues))

    params['locationAccuracy'] = location_accuracy

    _set_context(params, context)

    task = 'ComputeSensorModel'
    job_values = _execute_task(gis, task, params)
    
    return job_values["result"]["url"]

###################################################################################################
## Alter processing states
###################################################################################################
def alter_processing_states(image_collection, new_states, *, gis=None, **kwargs):
    '''
    Alter the processing states of the image collection.
    The states are stored as key property "Orthomapping". 
    The content of the state is a dictionary including 
    several properties which can be set based on the process 
    done on the image collection. 

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, This is the image collection that will be adjusted.

                           The image_collection can be a portal Item or an image service URL or URI
                           
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    new_states             Required dictionary. The state to set on the image_collection

                           This a dictionary of states that should be set on the image collection
                           The new states that can be set on the image collection are:
                           blockadjustment, dem, gcp, seamlines, colorcorrection, adjust_index, imagetype

                           Example:
                           {"blockadjustment": "raw",
                            "dem": "Dense_Natual_Neighbor",
                            "seamlines":"VORONOI",
                            "colorcorrection":"SingleColor",
                            "imagetype": "UAV/UAS",
                            "adjust_index": 0}
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================

    :return:
        The result will be the newly set states dictionary

    '''

    gis = arcgis.env.active_gis if gis is None else gis
        
    params = {}

    _set_image_collection_param(gis, params, image_collection)

    newStatesAllowedValues = ['blockadjustment', 'dem', 'gcp', 'seamlines', 'colorcorrection', 'adjust_index', 'imagetype']

    for key in new_states:
        if not key in newStatesAllowedValues:
            raise RuntimeError('new_states can only be one of the following: ' + str(newStatesAllowedValues))

    params['newStates'] = json.dumps(new_states)
    task = 'AlterProcessingStates'
    job_values = _execute_task(gis, task, params)
    if "processingStates" in job_values:
        if isinstance(job_values["processingStates"], dict):
            return job_values["processingStates"]
        elif isinstance(job_values["processingStates"], str):
            processing_states = job_values['processingStates'].replace("'",'"')
            processing_states=json.loads( processing_states.replace('u"','"'))
            return processing_states
 

###################################################################################################
## Get processing states
###################################################################################################
def get_processing_states(image_collection, *, gis=None, **kwargs):
    '''
    Retrieve the processing states of the image collection

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, This is the image collection that will be adjusted.

                           The image_collection can be a portal Item or an image service URL or URI
                            
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================

    :return:
        The result will be the newly set states dictionary

    '''

    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    _set_image_collection_param(gis, params, image_collection)

    task = 'GetProcessingStates'
    job_values = _execute_task(gis, task, params)
    return job_values["processingStates"]

"""
###################################################################################################
## Append control points
###################################################################################################
def append_control_points(image_collection, control_points, gis = None):
    '''
    Append additional ground control point sets to the image collection's control points. 
    A complete ground control point (GCP) set should have one ground control point 
    and multiple (more than 3) tie points.

    See http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/block-adjustment-for-mosaic-datasets.htm#ESRI_SECTION1_6676F2BB9A6B453E9EE1E00B42C4A5C1
    for more information about preparing control points and tie points for orthomapping

    Parameters
    ----------
    image_collection    :   Required, the input image collection on which to compute
                            the sensor model.

                            The image_collection can be a portal Item or an image service URL
                            
                            The image_collection must exist.

    control_points      :   Required, a list of control point objects.

                            A control point object is a dictionary with key-value
                            pairs as described below:

                            The schema of control points follows the schema 
                            of the mosaic dataset control point table. The following are
                            required when defining control points:
                            - The control points must contain a Point geometry object 

                            - There must be one attribute set, describing the attributes of the control point. 

                              The control point attributes is a dictionary that must contain the following 
                              key-value pairs:

                              -- imageID (int) - Image identification using the ObjectID from the mosaic dataset footprint table.

                              -- pointID (int) - The ID of the point within the control point table

                              -- type (int)    - The type of the control point as determined by its numeric value
                                                 1: Tie Point 
                                                 2: Ground Control Point.
                                                 3: Check Point

                              -- status (int)  - The status of the point. A value of 0 indicates that the point will
                                                 not be used in computation. A non-zero value indicates otherwise.

                            Example:
                            {"geometry": {
                                "x":-118.15,"y":33.80,"z":10.0,
                                "spatialReference":{"wkid":4326}},  
                                "attributes": {
                                   "imageID": 22,
                                   "pointID": 2, 
                                   "type": 2,
                                   "status": 1, 
                                 },
                            <more points>
                            }

    '''
    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    _set_image_collection_param(params, image_collection)

    params['controlPoints'] = json.dumps(control_points)
    task = 'AppendControlPoints'
    _execute_task(gis, task, None)
    return
"""
###################################################################################################
## Match control points
###################################################################################################
def match_control_points(image_collection, control_points, similarity=None, context=None, *, gis=None, **kwargs):
    '''
    The match_control_points is a function that takes a collection of ground control points
    as input (control points to be specified as a list of dictionary objects), and each of the 
    ground control points needs at least one matching tie point in the control point sets. 
    The function will compute the remaining matching tie points for all control point sets.
    
    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, the input image collection that will be adjusted.

                           The image_collection can be a portal Item or an image service URL or a URI
                            
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    control_points         Required, a list of control point sets objects.

                           The schema of control points follows the schema 
                           of the mosaic dataset control point table. 

                           The control point object should contain the point geometry, pointID, type, status and the
                           imagePoints. (the imagePoints attribute inside the control points object lists the imageIDs)

                           -- pointID (int) - The ID of the point within the control point table.

                           -- type (int)    - The type of the control point as determined by its numeric value
                                                 1: Tie Point 
                                                 2: Ground Control Point.
                                                 3: Check Point

                           -- status (int)  - The status of the point. A value of 0 indicates that the point will
                                                 not be used in computation. A non-zero value indicates otherwise.

                           

                           -- imageID (int) - Image identification using the ObjectID from the mosaic dataset footprint table.

                           Example:
                           [{
                           "status": 1,
                           "type": 2,
                           "x": -117.0926538,
                           "y": 34.00704253,
                           "z": 634.2175,
                           "spatialReference": {
                               "wkid": 4326
                           }, // default WGS84
                           "imagePointSpatialReference": {}, // default ICS
                           "pointId": 1,
                           "xyAccuracy": "0.008602325",
                           "zAccuracy": "0.015",
                           "imagePoints": [{
                               "imageID": 1,
                               "x": 2986.5435987557084,
                               "y": -2042.5193648409431,
                               "u": 3057.4580682832734,
                               "v": -1909.1506872159698
                           },
                           {
                               "imageID": 2,
                               "x": 1838.2814361401108,
                               "y": -2594.5280063817972,
                               "u": 3059.4079724863363,
                               "v": -2961.292545463305
                           },
                           {
                               "imageID": 12,
                               "x": 5332.855578204663,
                               "y": -2533.2805429751907,
                               "u": 614.2338676573158,
                               "v": -165.10836768947297
                           },
                           {
                               "imageID": 13,
                               "x": 4932.0895715254455,
                               "y": -1833.8401744114287,
                               "u": 616.9396928182223,
                               "v": -1243.1445126959693
                           }]
                           },
                           …
                           …
                           ] 
    ------------------     --------------------------------------------------------------------
    similarity             Optional string. Choose the tolerance level for your control point matching. 

                           Low- The similarity tolerance for finding control points will be low.
                           This option will produce the most control points, 
                           but some may have a higher level of error. 

                           Medium - The similarity tolerance for finding control points will be medium.
                           
                           High - The similarity tolerance for finding control points will be high. 
                           This option will produce the least number of control points, 
                           but each matching pair will have a lower level of error. This is the default. 
    ------------------     --------------------------------------------------------------------
    context                Optional dictionary.Additional settings such as the input control points 
                           spatial reference can be specified here. 

                           For example:
                           {"groundControlPointsSpatialReference": {"wkid": 3459}, "imagePointSpatialReference": {"wkid": 3459}}

                           Note: The ground control points spatial reference and image point spatial reference 
                           spatial reference set in the context parameter is to decide the returned point set's 
                           ground control points spatial reference and image point spatial reference. 
                           If these two parameters are not set here, the tool will use the spatial reference 
                           defined in the input point set. And if no spatial reference is defined in the point set,
                           then the default ground control points coordinates are in lon/lat and image points 
                           coordinates are in image coordinate system. 
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================

    :return:
        A dictionary object

    '''

    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    _set_image_collection_param(gis, params, image_collection)

    params['inputControlPoints'] = json.dumps(control_points)

    similarityAllowedValues = ['Low', 'Medium', 'High']
    if similarity is not None:
        if not similarity in similarityAllowedValues:
            raise RuntimeError('similarity can only be one of the following: '+str(similarityAllowedValues))
        params['similarity'] = similarity

    if context is not None:
        params['context'] = json.dumps(context)

    task = 'MatchControlPoints'
    job_values = _execute_task(gis, task, params)

    if job_values["result"] is not None:
        gptool_url = gis.properties.helperServices.orthoMapping.url
        gptool = arcgis.gis._GISResource(gptool_url, gis)
        result = gptool._con.post(job_values["result"]["url"],{},token=gptool._token)
    else:
        return job_values["result"]

    return result

###################################################################################################
## Color Correction
###################################################################################################
def color_correction(image_collection,
                  color_correction_method,
                  dodging_surface_type,
                  target_image=None,
                  context = None,
                   *, 
                   gis=None, 
                   **kwargs):
    '''
    Color balance the image collection. 
    Refer to the "Color Balance Mosaic Dataset" GP tool for 
    documentation on color balancing mosaic datasets.
    http://pro.arcgis.com/en/pro-app/tool-reference/data-management/color-balance-mosaic-dataset.htm

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    image_collection                         Required. This is the image collection that will be adjusted.

                                             The image_collection can be a portal Item or an image service URL or a URI
                            
                                             The image_collection must exist.
    ------------------------------------     --------------------------------------------------------------------
    color_correction_method                  Required string. This is the method that will be used for color
                                             correction computation. The available options are:

                                             Dodging-Change each pixel's value toward a target color. 
                                             With this technique, you must also choose 
                                             the type of target color surface, which 
                                             affects the target color. Dodging tends 
                                             to give the best result in most cases. 

                                             Histogram-Change each pixel's value according
                                             to its relationship with a target histogram. 
                                             The target histogram can be derived from
                                             all of the rasters, or you can specify a
                                             raster. This technique works well when 
                                             all of the rasters have a similar histogram.
                                    
                                             Standard_Deviation-Change each of the pixel's
                                             values according to its relationship with the
                                             histogram of the target raster, within one 
                                             standard deviation. The standard deviation can be
                                             calculated from all of the rasters in the mosaic
                                             dataset, or you can specify a target raster. 
                                             This technique works best when all of the 
                                             rasters have normal distributions.
    ------------------------------------     --------------------------------------------------------------------
    dodging_surface_type                     Required string.When using the Dodging balance method, 
                                             each pixel needs a target color, which is determined by 
                                             the surface type.

                                             Single_Color-Use when there are only a small
                                             number of raster datasets and a few different
                                             types of ground objects. If there are too many
                                             raster datasets or too many types of ground 
                                             surfaces, the output color may become blurred. 
                                             All the pixels are altered toward a single 
                                             color point-the average of all pixels. 
                                    
                                             Color_Grid- Use when you have a large number
                                             of raster datasets, or areas with a large 
                                             number of diverse ground objects. Pixels 
                                             are altered toward multiple target colors, 
                                             which are distributed across the mosaic dataset. 

                                             First_Order- This technique tends to create a 
                                             smoother color change and uses less storage in
                                             the auxiliary table, but it may take longer to
                                             process compared to the color grid surface. 
                                             All pixels are altered toward many points obtained
                                             from the two-dimensional polynomial slanted plane. 

                                             Second_Order-This technique tends to create a 
                                             smoother color change and uses less storage in
                                             the auxiliary table, but it may take longer to
                                             process compared to the color grid surface. 
                                             All input pixels are altered toward a set of 
                                             multiple points obtained from the two-dimensional 
                                             polynomial parabolic surface. 

                                             Third_Order-This technique tends to create a 
                                             smoother color change and uses less storage in
                                             the auxiliary table, but it may take longer to
                                             process compared to the color grid surface. 
                                             All input pixels are altered toward multiple 
                                             points obtained from the cubic surface.
    ------------------------------------     --------------------------------------------------------------------
    target_image                             Optional. The image service you want to use to color balance 
                                             the images in the image collection.
                                             It can be a portal Item or an image service URL or a URI
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dictionary. It contains additional settings that allows
                                             users to customize the statistics computation settings.

                                             Example:
                                             {"skipRows": 10, "skipCols": 10, "reCalculateStats": "OVERWRITE"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. the GIS on which this tool runs. If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
        The imagery layer url

    '''

    gis = arcgis.env.active_gis if gis is None else gis

    params = {}
    context = {}
    _set_image_collection_param(gis,params, image_collection)

    color_correction_allowed_values = ['DODGING', 'HISTOGRAM', 'STANDARD_DEVIATION']

    if not color_correction_method.upper() in color_correction_allowed_values:
            raise RuntimeError('similarity can only be one of the following: '+ str(color_correction_allowed_values))        
    params['colorCorrectionMethod'] = color_correction_method

    dodging_surface_type_allowed_values = ['SINGLE_COLOR', 'COLOR_GRID', 'FIRST_ORDER','SECOND_ORDER','THIRD_ORDER']
    if not dodging_surface_type.upper() in dodging_surface_type_allowed_values:
            raise RuntimeError('similarity can only be one of the following: '+ str(dodging_surface_type_allowed_values))
    params['dodgingSurface'] = dodging_surface_type

    if target_image is not None:
        if isinstance(target_image, str):
            if 'http:' in target_image or 'https:' in target_image:
                params['targetImage'] = json.dumps({ 'url' : target_image })
            else:
                params['targetImage'] = json.dumps({ 'uri' : target_image })
        elif isinstance(target_image, Item):
                params['targetImage'] = json.dumps({ "itemId" : target_image.itemid })
        else:
            raise TypeError("target_image should be a string (url or uri) or Item")
    
    _set_context(params, context)        
    
    task = 'ComputeColorCorrection'
    job_values = _execute_task(gis, task, params)


    return job_values["result"]["url"]



###################################################################################################
## Compute Control Points
###################################################################################################
def compute_control_points(image_collection, reference_image=None, image_location_accuracy="High", context = None, *, gis=None, **kwargs):
    '''
    This service tool is used for computing matching control points between images
    within an image collection and/or matching control points between the image 
    collection images and the reference image.
    http://pro.arcgis.com/en/pro-app/tool-reference/data-management/compute-control-points.htm
    
    ====================================    ====================================================================
    **Argument**                            **Description**
    ------------------------------------    --------------------------------------------------------------------
    image_collection                        Required. This is the image collection that will be adjusted.

                                            The image_collection can be a portal Item or an image service URL or a URI
                            
                                            The image_collection must exist.
    ------------------------------------    --------------------------------------------------------------------
    reference_image                         This is the reference image service that can be used to generate ground control 
                                            points set with the image service. 
                                            It can be a portal Item or an image service URL or a URI
    ------------------------------------    --------------------------------------------------------------------
    image_location_accuracy                 Optional string. This option allows users to specify the location accuracy of the  
                                            imagery.
                                            VERYLOW, LOW, MEDIUM, HIGH

                                            LOW-Images have a large shift and a large rotation (> 5 degrees).
                                                The SIFT algorithm will be used in the point matching computation. 

                                            MEDIUM-Images have a medium shift and a small rotation (<5 degrees).
                                                   The Harris algorithm will be used in the point matching computation. 

                                            HIGH-Images have a small shift and a small rotation.
                                                 The Harris algorithm will be used in the point matching computation. 

                                            Default is HIGH
    ------------------------------------    --------------------------------------------------------------------
    context                                 Optional dictionary. Context contains additional environment settings that affect 
                                            output control points generation.
                                            {
                                            "pointSimilarity":"MEDIUM",
                                            "pointDensity": "MEDIUM",
                                            "pointDistribution": "RANDOM"
                                            }
    ------------------------------------    --------------------------------------------------------------------
    gis                                     Optional GIS. the GIS on which this tool runs. If not specified, the active GIS is used.
    ====================================    ====================================================================

    :return:
        The imagery layer url

    '''
    gis = arcgis.env.active_gis if gis is None else gis

    params = {}
    context = {}
    _set_image_collection_param(gis, params, image_collection)

    if reference_image is not None:
        if isinstance(reference_image, str):
            if 'http:' in reference_image or 'https' in reference_image:
                params['referenceImage'] = json.dumps({ 'url' : reference_image })
            else:
                params['referenceImage'] = json.dumps({ 'uri' : reference_image })
        elif isinstance(reference_image, Item):
                params['referenceImage'] = json.dumps({ "itemId" : reference_image.itemid })
        else:
            raise TypeError("reference_image should be a string (url or uri) or Item")

    params["imageLocationAccuracy"]=image_location_accuracy

    _set_context(params, context)  

    task = 'ComputeControlPoints'
    job_values = _execute_task(gis, task, params)

    return job_values["result"]

###################################################################################################
## Compute Seamlines
###################################################################################################
def compute_seamlines(image_collection,
                      seamlines_method,
                      context = None,
                      *, 
                      gis=None, 
                      **kwargs):
    '''
    Compute seamlines on the image collection. This service tool is used to compute
    seamlines for the image collection, usually after the image collection has been
    block adjusted. Seamlines are helpful for generating the seamless mosaicked 
    display of overlapped images in image collection. The seamlines are computed
    only for candidates that will eventually be used for generating the result
    ortho-mosaicked image.  
    http://pro.arcgis.com/en/pro-app/tool-reference/data-management/build-seamlines.htm

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, the input image collection that will be adjusted.
                           The image_collection can be a portal Item or an image service URL or a URI
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    seamlines_method       Required string. These are supported methods for generated seamlines for the image collection.
    
                            VORONOI-Generate seamlines using the area Voronoi diagram. 

                            DISPARITY-Generate seamlines based on the disparity images of stereo pairs.
                            
                            GEOMETRY - Generate seamlines for overlapping areas based on the intersection 
                            of footprints. Areas with no overlapping imagery will merge the footprints. 

                            RADIOMETRY - Generate seamlines based on the spectral patterns of features 
                            within the imagery.

                            EDGE_DETECTION - Generate seamlines over intersecting areas based on the 
                            edges of features in the area.

                            This method can avoid seamlines cutting through buildings. 
    ------------------     --------------------------------------------------------------------
    context                Optional dictionary. Context contains additional settings that allows users to customize
                           the seamlines generation. 
                           Example:
                           {"minRegionSize": 100,
                           "pixelSize": "",
                           "blendType": "Both",
                           "blendWidth": null,
                           "blendUnit": "Pixels",
                           "requestSizeType": "Pixels",
                           "requestSize": 1000,
                           "minThinnessRatio": 0.05,
                           "maxSilverSize": 20
                           }

                           Allowed keys are:
                           "minRegionSize", "pixelSize", "blendType", "blendWidth", 
                           "blendUnit", "requestSizeType", "requestSize", 
                           "minThinnessRatio", "maxSilverSize"
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================

    :return:
        The Imagery layer url

    '''
    gis = arcgis.env.active_gis if gis is None else gis

    params = {}
    context={}
    _set_image_collection_param(gis, params, image_collection)

    contextAllowedValues= {"minRegionSize", "pixelSize", "blendType", "blendWidth", 
                           "blendUnit", "requestSizeType", "requestSize", 
                           "minThinnessRatio", "maxSilverSize"
                           }

    seamlines_method_allowed_values = ['VORONOI', 'DISPARITY','GEOMETRY', 'RADIOMETRY', 'EDGE_DETECTION']
    if not seamlines_method.upper() in seamlines_method_allowed_values:
            raise RuntimeError('similarity can only be one of the following: '+str(seamlines_method_allowed_values))
    params['seamlinesMethod'] = seamlines_method

    _set_context(params, context)

    task = 'ComputeSeamlines'
    job_values = _execute_task(gis, task, params)

    return job_values["result"]["url"]


###################################################################################################
## Edit control points
###################################################################################################
def edit_control_points(image_collection, control_points, *, gis=None, **kwargs):
    '''
    This service can be used to append additional ground control point sets to
    the image collection's control points. It is recommended that a ground control point (GCP) set 
    should contain one ground control point and multiple tie points. 
    The service tool can also be used to edit tie point sets. 
    The input control points dictionary will always replace the points in the tie points
    table if the point IDs already exist. 
   
    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required.
                           The image_collection can be a portal Item or an image service URL or a URI
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    control_points         Required, a list of control point sets objects.

                           The schema of control points follows the schema 
                           of the mosaic dataset control point table. 

                           The control point object should contain the point geometry, pointID, type, status and the
                           imagePoints. (the imagePoints attribute inside the control points object lists the imageIDs)

                           -- pointID (int) - The ID of the point within the control point table.

                           -- type (int)    - The type of the control point as determined by its numeric value
                                                 1: Tie Point 
                                                 2: Ground Control Point.
                                                 3: Check Point

                           -- status (int)  - The status of the point. A value of 0 indicates that the point will
                                                 not be used in computation. A non-zero value indicates otherwise.

                           

                           -- imageID (int) - Image identification using the ObjectID from the mosaic dataset footprint table.

                           Example:
                           [{
                           "status": 1,
                           "type": 2,
                           "x": -117.0926538,
                           "y": 34.00704253,
                           "z": 634.2175,
                           "spatialReference": {
                               "wkid": 4326
                           }, // default WGS84
                           "imagePointSpatialReference": {}, // default ICS
                           "pointId": 1,
                           "xyAccuracy": "0.008602325",
                           "zAccuracy": "0.015",
                           "imagePoints": [{
                               "imageID": 1,
                               "x": 2986.5435987557084,
                               "y": -2042.5193648409431,
                               "u": 3057.4580682832734,
                               "v": -1909.1506872159698
                           },
                           {
                               "imageID": 2,
                               "x": 1838.2814361401108,
                               "y": -2594.5280063817972,
                               "u": 3059.4079724863363,
                               "v": -2961.292545463305
                           },
                           {
                               "imageID": 12,
                               "x": 5332.855578204663,
                               "y": -2533.2805429751907,
                               "u": 614.2338676573158,
                               "v": -165.10836768947297
                           },
                           {
                               "imageID": 13,
                               "x": 4932.0895715254455,
                               "y": -1833.8401744114287,
                               "u": 616.9396928182223,
                               "v": -1243.1445126959693
                           }]
                           },
                           …
                           …
                           ] 


    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================

    :return:
        The Imagery layer url

    '''
    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    _set_image_collection_param(gis, params, image_collection)

    params['inputControlPoints'] = json.dumps(control_points)

    task = 'EditControlPoints'
    job_values = _execute_task(gis, task, params)

    return job_values["result"]["url"]

###################################################################################################
## Generate DEM
###################################################################################################
def generate_dem(image_collection,
                 out_dem,
                 cell_size,
                 surface_type,
                 matching_method = None,
                 context = None,
                 *,
                 gis = None,
                 **kwargs):
    '''
    Generate a DEM from the image collection. Refer to "Interpolate From Point Cloud"
    GP tool for more documentation
    http://pro.arcgis.com/en/pro-app/tool-reference/data-management/interpolate-from-point-cloud.htm
    
    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required. The input image collection that will be used
                           to generate the DEM from.
                           The image_collection can be a portal Item or an image service URL or a URI
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    out_dem                This is the output digital elevation model image url or name. 
                           Like Raster Analysis services, the service can be an existing multi-tenant service URL 
                           or this service tool can create it. 
                           It can be a url, uri, portal item, string representing the name of output image service 
    ------------------     --------------------------------------------------------------------
    cell_size              Required, The cell size of the output raster dataset. This is a single numeric input. 
                           Rectangular cell size such as {"x": 10, "y": 10} is not supported. 
                           The cell size unit will be the unit used by the image collection's spatial reference.
    ------------------     --------------------------------------------------------------------
    surface_type           Required string. Create a digital terrain model or a digital surface model. Refer
                           to "surface_type" parameter of the GP tool.
                           The available choices are:

                           DTM - Digital Terrain Model, the elevation is only the elevation of the bare earth, not including structures above the surface.

                           DSM - Digital Surface Model, the elevation includes the structures above the surface, for example, buildings, trees, bridges.
    ------------------     --------------------------------------------------------------------
    matching_method        Optional string. The method used to generate 3D points. 

                           ETM-A feature-based stereo matching that uses the Harris operator to 
                           detect feature points. It is recommended for DTM generation.  

                           SGM- Produces more points and more detail than the ETM method. It is 
                           suitable for generating a DSM for urban areas. This is more 
                           computationally intensive than the ETM method1.  

                           MVM (Multi-view image matching (MVM) - is based on the SGM matching method followed by a fusion step in which 
                           the redundant depth estimations across single stereo model are merged. 
                           It produces dense 3D points and is computationally efficient

                           References:  
                           Heiko Hirschmuller et al., "Memory Efficient Semi-Global Matching," 
                           ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial 
                           Information Sciences, Volume 1-3, (2012): 371-376. 

                           Refer to the documentation
                           of "matching_method" parameter of the "Generate Point Cloud"
                           GP tool @ http://pro.arcgis.com/en/pro-app/tool-reference/data-management/generate-point-cloud.htm
    ------------------     --------------------------------------------------------------------
    context                Optional dictionary. Additional allowed point cloud generation parameter and DEM 
                           interpolation parameter can be assigned here.  
                           For example: 
                           Point cloud generation parameters -  
                           {"maxObjectSize": 50, 
                           "groundSpacing": None, 
                           "minAngle": 10, 
                           "maxAngle": 70, 
                           "minOverlap": 0.6, 
                           "maxOmegaPhiDif": 8, 
                           "maxGSDDif": 2, 
                           "numImagePairs": 2, 
                           "adjQualityThreshold": 0.2, 
                           "regenPointCloud": False 
                           } 

                           DEM interpolation parameters -  
                           {"method": "TRIANGULATION", 
                           "smoothingMethod": "GAUSS5x5", 
                           "applyToOrtho": True, 
                           "fillDEM": "https://...." 
                           } 
 
                           Note:  
                           The "applyToOrtho" flag can apply the generated DEM back into the 
                           mosaic dataset's geometric function to achieve more accurate 
                           orthorectification result.  
                           The "fillDEM" flag allows the user to specify an elevation service URL as 
                           background elevation to fill the area when elevation model pixels cannot be 
                           interpolated from the point cloud.  
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================

    :return:
        The DEM layer item

    '''

    gis = arcgis.env.active_gis if gis is None else gis

    task = 'GenerateDEM'

    contextAllowedValues= ["maxObjectSize", "groundSpacing", "minAngle", "maxAngle", "minOverlap", "maxOmegaPhiDif", 
                            "maxGSDDif", "numImagePairs", "adjQualityThreshold", "method", "smoothingMethod", "applyToOrtho"]
    context = {}
    params = {}
    folder = None
    folderId = None
    _set_image_collection_param(gis, params, image_collection)

    if isinstance(out_dem, Item):
        params["outputDEM"] = json.dumps({"itemId": out_dem.itemid})
    elif isinstance(out_dem, str):
        if ("/") in out_dem or ("\\") in out_dem:
            if 'http:' in out_dem or 'https:' in out_dem:
                params['outputDEM'] = json.dumps({ 'url' : out_dem })
            else:
                params['outputDEM'] = json.dumps({ 'uri' : out_dem })
        else:
            result = gis.content.search("title:"+str(out_dem), item_type = "Imagery Layer")
            if len(result)>0:
                result = result[0]
                if result is not None:
                    params["outputDEM"]= json.dumps({"itemId": result.itemid})
            else:
                doesnotexist = gis.content.is_service_name_available(out_dem, "Image Service") 
                if doesnotexist:
                    if kwargs is not None:
                        if "folder" in kwargs:
                            folder = kwargs["folder"]
                    if folder is not None:
                        if isinstance(folder, dict):
                            if "id" in folder:
                                folderId = folder["id"]
                                folder=folder["title"]
                        else:
                            owner = gis.properties.user.username
                            folderId = gis._portal.get_folder_id(owner, folder)
                        if folderId is None:
                            folder_dict = gis.content.create_folder(folder, owner)
                            folder = folder_dict["title"]
                            folderId = folder_dict["id"]
                        params["outputDEM"] = json.dumps({"serviceProperties": {"name" : out_dem}, "itemProperties": {"folderId" : folderId}})
                    else:
                        params["outputDEM"] = json.dumps({"serviceProperties": {"name" : out_dem}})
              


    params['cellSize'] = cell_size

    surfaceTypeAllowedValues = ['DTM', 'DSM']
    if not surface_type in surfaceTypeAllowedValues:
        raise RuntimeError('surface_type can only be one of the following: '+ str(surfaceTypeAllowedValues))
    params['surfaceType'] = surface_type
    
    if matching_method is not None:
        params['matchingMethod'] = matching_method

    _set_context(params, context)  
    
    job_values = _execute_task(gis, task, params)

    output_service= gis.content.get(job_values["result"]["itemId"])

    return  output_service 


###################################################################################################
## Generate orthomosaic
###################################################################################################
def generate_orthomosaic(image_collection,
                         out_ortho,
                         regen_seamlines=True,
                         recompute_color_correction=True,
                         context=None,
                         *,
                         gis=None,
                         **kwargs):
    '''
    Function can be used for generating single ortho-rectified mosaicked image from image collection after 
    the block adjustment.  
    
    ===================================    ====================================================================
    **Argument**                           **Description**
    -----------------------------------    --------------------------------------------------------------------
    image_collection                       Required. The input image collection that will be used
                                           to generate the ortho-mosaic from.
                                           The image_collection can be a portal Item or an image service URL or a URI
                                           The image_collection must exist.
    -----------------------------------    --------------------------------------------------------------------
    out_ortho                               Required. This is the output ortho-mosaicked image converted from the image 
                                            collection after the block adjustment.   
                                            It can be a url, uri, portal item, string representing the name of output image 
                                            service  
    -----------------------------------    --------------------------------------------------------------------
    regen_seamlines                        Optional, boolean. 

                                           Choose whether to apply seamlines before the orthomosaic image generation or not. 
                                           The seamlines will always be regenerated if this parameter is set to True. 
                                           The user can set the seamline options through the context parameter. 
                                           If the seamline generation options are not set, the default will be used.  

                                            Default value is True
    -----------------------------------    --------------------------------------------------------------------
    recompute_color_correction              Optional, boolean.
                                            Choose whether to apply color correction settings to the output ortho-image or not. 
                                            Color correction will always be recomputed if this option is set to True. 
                                            The user can configure the compute color correction settings through the context parameter. 
                                            If there is no color collection setting, the default will be used.  

                                            Default value is True
    -----------------------------------    --------------------------------------------------------------------
    context                                Optional dictionary. Context contains additional environment settings that affect output 
                                           image. The supported environment settings for this tool are:

                                           1. Output Spatial Reference (outSR)-the output features will
                                              be projected into the output spatial reference.

                                           2. Extent (extent) - extent that would clip or expand the output image 
    -----------------------------------    --------------------------------------------------------------------
    gis                                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ===================================    ====================================================================

    :return:
        The Orthomosaicked Imagery layer item

    '''
    gis = arcgis.env.active_gis if gis is None else gis

    task = 'GenerateOrthomosaic'

    params = {}
    folder = None
    folderId = None
    _set_image_collection_param(gis, params, image_collection)
        
    if isinstance(out_ortho, Item):
        params["outputOrthoImage"] = json.dumps({"itemId": out_ortho.itemid})
    elif isinstance(out_ortho, str):
        if ("/") in out_ortho or ("\\") in out_ortho:
            if 'http:' in out_ortho or 'https:' in out_ortho:
                params['outputOrthoImage'] = json.dumps({ 'url' : out_ortho })
            else:
                params['outputOrthoImage'] = json.dumps({ 'uri' : out_ortho })
        else:
            result = gis.content.search("title:"+str(out_ortho), item_type = "Imagery Layer")
            if len(result) > 0:
                result = result[0]
                if result is not None:
                    params["outputOrthoImage"]= json.dumps({"itemId": result.itemid})
            else:
                doesnotexist = gis.content.is_service_name_available(out_ortho, "Image Service") 
                if doesnotexist:
                    if kwargs is not None:
                        if "folder" in kwargs:
                            folder = kwargs["folder"]
                    if folder is not None:
                        if isinstance(folder, dict):
                            if "id" in folder:
                                folderId = folder["id"]
                                folder=folder["title"]
                        else:
                            owner = gis.properties.user.username
                            folderId = gis._portal.get_folder_id(owner, folder)
                        if folderId is None:
                            folder_dict = gis.content.create_folder(folder, owner)
                            folder = folder_dict["title"]
                            folderId = folder_dict["id"]
                        params["outputOrthoImage"] = json.dumps({"serviceProperties": {"name" : out_ortho}, "itemProperties": {"folderId" : folderId}})
                    else:
                        params["outputOrthoImage"] = json.dumps({"serviceProperties": {"name" : out_ortho}})
            

    if regen_seamlines is not None:
        if not isinstance(regen_seamlines, bool):
            raise TypeError("The 'regen_seamlines' parameter must be a boolean")
        params['regenSeamlines'] = regen_seamlines

    if recompute_color_correction is not None:
        if not isinstance(recompute_color_correction, bool):
            raise TypeError("The 'recompute_color_correction' parameter must be a boolean")
        params['recomputeColorCorrection'] = recompute_color_correction    

    _set_context(params, context)   
    
    job_values = _execute_task(gis, task, params)

    output_service= gis.content.get(job_values["result"]["itemId"])

    return  output_service 


###################################################################################################
## Generate report
###################################################################################################
def generate_report(image_collection, report_format="PDF", *, gis=None, **kwargs):
    """
    This function is used to generate orthomapping report with image collection 
    that has been block adjusted. The report would contain information about 
    the quality of the adjusted images, the distribution of the control points, etc.
    The output of this service tool is a downloadable html page. 

    ===================    ====================================================================
    **Argument**           **Description**
    -------------------    --------------------------------------------------------------------
    image_collection       Required. the input image collection that should be
                           used to generate a report from.
                           The image_collection can be a portal Item or an image service URL or a URI
                           The image_collection must exist.
    -------------------    --------------------------------------------------------------------
    report_format          Type of the format to be generated. Possible PDF, HTML. Default - PDF
    -------------------    --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ===================    ====================================================================

    :return:
        The URL of a single html webpage that is a formatted orthomapping report

    """
    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    _set_image_collection_param(gis, params, image_collection)

    report_format_allowed_values = ['PDF', 'HTML']
    if not report_format in report_format_allowed_values:
        raise RuntimeError('report_format can only be one of the following: '+ str(report_format_allowed_values))
    params['reportFormat'] = report_format

    task = 'GenerateReport'
    job_values = _execute_task(gis, task, params)

    return job_values["outReport"]["url"]

###################################################################################################
## query camera info
###################################################################################################
def query_camera_info(camera_query=None,
                      *, 
                      gis=None,
                      **kwargs):
    ''' 
    This service tool is used to query specific or the entire digital camera 
    database. The digital camera database contains the specs
    of digital camera sensors that were used to capture drone images. 

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    camera_query           Required String. This is a SQL query statement that can 
                           be used to filter a portion of the digital camera
                           database.
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================


    :return:
        Data Frame representing the camera database

    '''
    import pandas as pd
    import numpy as np

    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    if camera_query is not None:
        if not isinstance(camera_query, str):
            raise TypeError("The 'camera_query' parameter must be a string")
        params['query'] = camera_query
   
    task = 'QueryCameraInfo'
    job_values = _execute_task(gis, task, params)
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(np.array(job_values["outputCameraInfo"]["content"]),columns = ["Maker","Model", "Focal Length", "Columns", "Rows","Pixel Size"])
    display(df)
    return df
    #return job_values["outputCameraInfo"]["content"]

###################################################################################################
## query control points
###################################################################################################
def query_control_points(image_collection,
                         query,
                         *, 
                         gis=None, 
                         **kwargs):
    '''
    Query for control points in an image collection. It allows users to query 
    among certain control point sets that has ground control points inside.

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, the input image collection on which to query 
                           the the control points.

                           The image_collection can be a portal Item or an image service URL or a URI.
                           
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    query                  Required string. a SQL statement used for querying the point;

                           e.g. "pointID > 100"
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================


    :return:
        A dictionary object

    '''
    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    _set_image_collection_param(gis, params, image_collection)

    if not isinstance(query, str):
        raise TypeError("The 'query' parameter must be a string")

    params['where'] = query

    task = 'QueryControlPoints'
    job_values = _execute_task(gis, task, params)

    if job_values["outControlPoints"] is not None:
        gptool_url = gis.properties.helperServices.orthoMapping.url
        gptool = arcgis.gis._GISResource(gptool_url, gis)
        result = gptool._con.post(job_values["outControlPoints"]["url"],{},token=gptool._token)
    else:
        return job_values["outControlPoints"]

    return result

###################################################################################################
## Reset image collection
###################################################################################################
def reset_image_collection(image_collection,
                            *, 
                            gis=None, 
                            **kwargs):
    '''
    Reset the image collection. It is used to reset the image collection to its 
    original state. The image collection could be adjusted during the orthomapping 
    workflow and if the user is not satisfied with the result, they will be able 
    to clear any existing adjustment settings and revert the images back to 
    un-adjusted state

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, the input image collection to reset
                           The image_collection can be a portal Item or an image service URL or a URI.
                           
                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ==================     ====================================================================

    :return:
        A boolean indicating whether the reset was successful or not

    '''
    gis = arcgis.env.active_gis if gis is None else gis

    params = {}

    _set_image_collection_param(gis, params, image_collection)

    task = 'ResetImageCollection'
    job_values = _execute_task(gis, task, params)
    return job_values["result"]


def compute_spatial_reference_factory_code(latitude, longitude): 
    """
    Computes spatial reference factory code. This value may be used as out_sr value in create image collection function

    Parameters
    ----------
    latitude : latitude value in decimal degress that will be used to compute UTM zone
    longitude : longitude value in decimal degress that will be used to compute UTM zone

    Returns
    -------
    factory_code : spatial reference factory code
    """
    from math import isnan, fabs, floor
    zone = 0
    if (isnan(longitude) or isnan(latitude) or fabs(longitude) > 180.0 or fabs(latitude) > 90.0):
        raise RuntimeError("Incorrect latitude or longitude value")

    zone = floor((longitude + 180)/6) + 1
    if (latitude >= 56.0 and latitude < 64.0 and longitude >= 3.0 and longitude < 12.0):
        zone = 32;

    if (latitude >= 72.0 and latitude < 84.0):
        if  (longitude >= 0.0  and longitude <  9.0):
            zone = 31;
        elif (longitude >= 9.0  and longitude < 21.0):
            zone = 33;
        elif (longitude >= 21.0 and longitude < 33.0):
            zone = 35;
        elif (longitude >= 33.0 and longitude < 42.0): 
            zone = 37

    if(latitude>=0):
        srid = 32601
    else:
        srid = 32701

    factory_code = srid + zone -1

    return factory_code
