"""
These tools help you identify, quantify, and visualize spatial patterns in your data.

calculate_density takes known quantities of some phenomenon and spreads these quantities across the map.
find_hot_spots identifies statistically significant clustering in the spatial pattern of your data.
"""
import json as _json
from datetime import datetime as _datetime
import logging as _logging
import arcgis as _arcgis
from arcgis.features import FeatureSet as _FeatureSet
from arcgis.geoprocessing._support import _execute_gp_tool
from arcgis.geoprocessing import DataFile
from ._util import _id_generator, _feature_input, _set_context, _create_output_service

_log=_logging.getLogger(__name__)

_use_async=True

def _forest_based_regression(input_layer,
                            prediction_type,
                            explanatory_variables,
                            features_to_predict,
                            variable_predict,
                            explanatory_variable_match,
                            return_importance_table,
                            number_of_trees=100,
                            min_leaf_size=5,
                            max_tree_depth=5,
                            sample_size=100,
                            random_vars=3,
                            percentage_for_validation=30,
                            output_name=None,
                            gis=None):
    """
    Creates models and generates predictions using an adaptation of Leo Breiman's random forest
    algorithm, which is a supervised machine learning method. Predictions can be performed for both
    categorical variables (classification) and continuous variables (regression). Explanatory
    variables can take the form of fields in the attribute table of the training features. In
    addition to validation of model performance based on the training data, predictions can be made
    to another feature dataset.

    The following are examples:

        + Given data on occurrence of seagrass, as well as a number of environmental explanatory
          variables represented as both attributes which has been enriched using a multi-variable grid
          to calculate distances to factories upstream and major ports, future seagrass occurrence can
          be predicted based on future projections for those same environmental explanatory variables.
        + Suppose you have crop yield data at hundreds of farms across the country along with other
          attributes at each of those farms (number of employees, acreage, and so on). Using these
          pieces of data, you can provide a set of features representing farms where you don't have
          crop yield (but you do have all of the other variables), and make a prediction about crop
          yield.
        + Housing values can be predicted based on the prices of houses that have been sold in the
          current year. The sale price of homes sold along with information about the number of
          bedrooms, distance to schools, proximity to major highways, average income, and crime counts
          can be used to predict sale prices of similar homes.


    **Forest Based Classification and Regression is available at ArcGIS Enterprise 10.7.**

    ==========================   ===============================================================
    **Argument**                 **Description**
    --------------------------   ---------------------------------------------------------------
    input_layer                  required FeatureSet, The table, point, line or polygon features
                                 containing potential incidents.
    --------------------------   ---------------------------------------------------------------
    prediction_type              Specifies the operation mode of the tool. The tool can be run to
                                 train a model to only assess performance, or train a model and
                                 predict features. Prediction types are as follows:

                                    + Train - This is the default. A model will be trained, but
                                      no predictions will be generated. Use this option to
                                      assess the accuracy of your model before generating
                                      predictions. This option will output model diagnostics in
                                      the messages window and a chart of variable importance.
                                    + TrainAndPredict - Predictions or classifications will be
                                      generated for features. Explanatory variables must be
                                      provided for both the training features and the features
                                      to be predicted. The output of this option will be a
                                      feature service, model diagnostics, and an optional
                                      table of variable importance.
    --------------------------   ---------------------------------------------------------------
    features_to_predict          Optional dict. The variable from the `input_layer` parameter
                                 containing the values to be used to train the model, and a
                                 boolean denoting if it's categorical. This field contains known
                                 (training) values of the variable that will be used to predict
                                 at unknown locations.
    --------------------------   ---------------------------------------------------------------
    variable_predict             Required dict. The variable from the input_layer parameter
                                 containing the values to be used to train the model, and a
                                 boolean denoting if it's categorical. This field contains known
                                 (training) values of the variable that will be used to predict
                                 at unknown locations.

                                 REST scripting example: "variable_predict":[{"fieldName":"isSunny",
                                                         "categorical":true},{"fieldName":"isWeekend",
                                                         "categorical":true}, {"fieldName":"hoursOutside",
                                                         "categorical":false}]
    --------------------------   ---------------------------------------------------------------
    explanatory_variable_match   A list of fields representing the explanatory variables and a
                                 boolean values denoting if the fields are categorical. The
                                 explanatory variables help predict the value or category of the
                                 variable_predict. Use the categorical parameter for any
                                 variables that represent classes or categories (such as
                                 landcover or presence or absence). Specify the variable as
                                 true for any that represent classes or categories such as
                                 landcover or presence or absence and false if the variable is
                                 continuous.

                                 Syntax: [{"fieldName":"<explanatory field name>", "categorical":true},

                                    + fieldname is the name of the field in the inFeatures used
                                      to predict the variable_predict.
                                    + categorical is one of: true or false. A string field should
                                      always be true, and a continue value should always be set as false.


    --------------------------   ---------------------------------------------------------------
    return_importance_table      Optional List. A list of the explanatory_variables specified
                                 from the inFeatures and their corresponding fields from the
                                 features_to_predict. By default, if an eexplanatory_variables is
                                 not mapped, it will match to a field with the same name in the
                                 features_to_predict. This parameter is only used if there is a
                                 features_to_predict input. You do not need to use it if the
                                 names and types of the fields match between your two input
                                 datasets.

                                 Syntax: [{"predictionLayerField":"<field name>", "trainingLayerField": "<field nane>"},...]

    --------------------------   ---------------------------------------------------------------
    number_of_trees              Optional int. The number of trees to create in the forest model.
                                 More trees will generally result in more accurate model
                                 prediction, but the model will take longer to calculate. The
                                 default number of trees is 100.
    --------------------------   ---------------------------------------------------------------
    min_leaf_size                Optional int. The minimum number of observations required to
                                 keep a leaf (that is the terminal node on a tree without
                                 further splits). The default minimum for regression is 5 and
                                 the default for classification is 1. For very large data,
                                 increasing these numbers will decrease the run time of the
                                 tool.
    --------------------------   ---------------------------------------------------------------
    max_tree_depth               Optional int. The maximum number of splits that will be made
                                 down a tree. Using a large maximum depth, more splits will be
                                 created, which may increase the chances of overfitting the
                                 model. The default is data driven and depends on the number of
                                 trees created and the number of variables included.
    --------------------------   ---------------------------------------------------------------
    sample_size                  Optional int. Specifies the percentage of the input_layer used
                                 for each decision tree. The default is 100 percent of the data.
                                 Samples for each tree are taken randomly from two-thirds of the
                                 data specified.
    --------------------------   ---------------------------------------------------------------
    random_vars                  Optional Int. Specifies the number of explanatory variables
                                 used to create each decision tree.Each of the decision trees in
                                 the forest is created using a random subset of the explanatory
                                 variables specified. Increasing the number of variables used in
                                 each decision tree will increase the chances of overfitting
                                 your model particularly if there is one or a couple dominant
                                 variables. A common practice is to use the square root of the
                                 total number of explanatory variables (fields, distances, and
                                 rasters combined) if your variablePredict is numeric or divide
                                 the total number of explanatory variables (fields, distances,
                                 and rasters combined) by 3 if variable_predict is categorical.
    --------------------------   ---------------------------------------------------------------
    percentage_for_validation    Optional Int. Specifies the percentage (between 10 percent
                                 and 50 percent) of inFeatures to reserve as the test dataset
                                 for validation. The model will be trained without this random
                                 subset of data, and the observed values for those features will
                                 be compared to the predicted value. The default is 10 percent.
    --------------------------   ---------------------------------------------------------------
    output_trained_name          optional String, The task will create a feature service of the
                                 results. You define the name of the service.
    --------------------------   ---------------------------------------------------------------
    gis                          optional GIS, the GIS on which this tool runs. If not
                                 specified, the active GIS is used.
    ==========================   ===============================================================

    :returns:
       Output feature layer item


    """
    kwargs=locals()

    gis=_arcgis.env.active_gis if gis is None else gis
    url=gis.properties.helperServices.geoanalytics.url

    params={}
    for key, value in kwargs.items():
        if value is not None:
            params[key]=value

    if output_name is None:
        output_service_name='Forest Based Regression_' + _id_generator()
        output_name=output_service_name.replace(' ', '_')
    else:
        output_service_name=output_name.replace(' ', '_')

    output_service=_create_output_service(gis, output_name, output_service_name, 'Forest Based Classification And Regression')

    params['output_name'] = _json.dumps({
        "serviceProperties": {"name" : output_name, "serviceUrl" : output_service.url},
        "itemProperties": {"itemId" : output_service.itemid}})


    _set_context(params)

    param_db={
        "input_layer": (_FeatureSet, "inFeatures"),
        "prediction_type" : (str, "predictionType"),
        "features_to_predict" : (_FeatureSet, "featuresToPredict"),
        "variable_predict" : (dict, "variablePredict"),
        "explanatory_variables" : (list, "explanatoryVariables"),
        "explanatory_variable_match" : (list, "explanatoryVariableMatching"),
        "return_importance_table" : (bool, "returnVariableOfImportanceTable"),
        "number_of_trees" : (int, "numberOfTrees"),
        "min_leaf_size" : (int, "minimumLeafSize"),
        "sample_size" : (int, "sampleSize"),
        "random_vars" : (str, "randomVariables"),
        "percentage_for_validation" : (int, "percentageForValidation"),
        "output_name" : (str, "outputTrainedName"),
        "context": (str, "context"),
        #"outputTrainedName": (_FeatureSet, "Output Features"),
    }
    return_values=[
        {"name": "outputTrainedName", "display_name": "Output Features", "type": _FeatureSet},
        {"name" : "outputPredicted", "display_name" : "Output Predicted", "type" : _FeatureSet},
        {"name" : "variableOfImportance", "display_name" : "Variable of Importance", "type" : _FeatureSet}
    ]

    try:
        _execute_gp_tool(gis, "ForestBasedClassificationAndRegression", params, param_db, return_values, _use_async, url, True)
        return output_service
    except:
        output_service.delete()
        raise

    return

def _generalized_linear_regression(input_layer,
                                  features_to_predict,
                                  dependent_variable,
                                  explanatory_variables,
                                  regression_family="Continuous",
                                  generate_coef_table=False,
                                  variable_matching=None,
                                  dependent_mapping=None,
                                  output_name=None,
                                  gis=None):
    """
    This tool performs Generalized Linear Regression (GLR) to generate
    predictions or to model a dependent variable's relationship to a set of
    explanatory variables. This tool can be used to fit continuous
    (Gaussian/OLS), binary (logistic), and count (Poisson) models.

    The following are examples of the tool's utility:

        + What demographic characteristics contribute to high rates of public transportation usage?
        + Is there a positive relationship between vandalism and burglary?
        + Which variables effectively predict 911 call volume? Given future projections, what is the expected demand for emergency response resources?
        + What variables affect low birth rates?

    ==========================   ===============================================================
    **Argument**                 **Description**
    --------------------------   ---------------------------------------------------------------
    input_layer                  Required FeatureSet. The table, point, line or polygon features.
    --------------------------   ---------------------------------------------------------------
    features_to_predict          Required FeatureSet. A layer containing features representing
                                 locations where estimates should be computed. Each feature in
                                 this dataset should contain values for all the explanatory
                                 variables specified. The dependent variable for these features
                                 will be estimated using the model calibrated for the input
                                 layer data.

                                 Syntax: As described in Feature input, this parameter can be
                                         one of the following:

                                    + A URL to a feature service layer with an optional filter
                                      to select specific features
                                    + A URL to a big data catalog service layer with an
                                      optional filter to select specific features
                                    + A feature collection

    --------------------------   ---------------------------------------------------------------
    dependent_variable           Required String. The numeric field containing the observed
                                 values you want to model.
    --------------------------   ---------------------------------------------------------------
    explanatory_variables        Required String. One or more fields representing independent
                                 explanatory variables in your regression model.
    --------------------------   ---------------------------------------------------------------
    regression_family            Required String. This field specifies the type of data you are
                                 modeling.

                                 regression_family is one of the following:

                                    + Continuous - The dependent_variable is continuous. The
                                                   model used is Gaussian, and the tool performs
                                                   ordinary least squares regression.
                                    + Binary - The dependent_variable represents presence or
                                               absence. Values must be 0 (absence) or 1 (presence)
                                               values, or mapped to 0 and 1 values using the
                                               parameter.
                                    + Count - The dependent_variable is discrete and represents
                                              events, such as crime counts, disease incidents,
                                              or traffic accidents. The model used is Poisson
                                              regression.
    --------------------------   ---------------------------------------------------------------
    generate_coef_table          Optional Boolean. Determines if a table with coefficient values
                                 will be returned. By default, the coefficient table is not
                                 returned.
    --------------------------   ---------------------------------------------------------------
    variable_matching            Optional List. A list of the explanatoryVariables specified from
                                 the input_layer and their corresponding fields from the
                                 features_to_predict. By default, if an explanatory_variables is
                                 not mapped, it will match to a field with the same name in the
                                 features_to_predict. This parameter is only used if there is a
                                 features_to_predict input. You do not need to use it if the
                                 names and types of the fields match between your two input
                                 datasets.

                                 Syntax: [{"predictionLayerField":"<field name>",
                                          "trainingLayerField": "<field name>"},...]

                                    + predictionLayerField is the name of a field specified in the
                                      explanatory_variables parameter.
                                    + trainingLayerField is the field that will match to the field
                                      in the explanatory_variables parameter.

                                 REST scripting example:

                                 "variablePredict":[{"predictionLayerField":"isSunny",
                                                    "trainingLayerField": "isSunny2010"}]

    --------------------------   ---------------------------------------------------------------
    dependent_mapping            Optional List. A list representing the values used to map to 0
                                 (absence) and 1 (presence) for binary regression.

                                 Syntax: [{"value0":"<false value>"},{"value1":"<true value>"}]

                                    + value0 is the string that will be used to represent 0
                                      (absence values).
                                    + value1 is the string that will be used to represent 1
                                      (presence values).

                                 REST scripting example: [{"value0":"Fail"},{"value1":"Pass"}]
    --------------------------   ---------------------------------------------------------------
    output_name                  Optional String. The task will create a feature service of the
                                 results. You define the name of the service.
    --------------------------   ---------------------------------------------------------------
    gis                          optional GIS, the GIS on which this tool runs. If not
                                 specified, the active GIS is used.
    ==========================   ===============================================================
    """
    kwargs=locals()

    gis=_arcgis.env.active_gis if gis is None else gis
    url=gis.properties.helperServices.geoanalytics.url

    params={}
    for key, value in kwargs.items():
        if value is not None:
            params[key]=value

    if output_name is None:
        output_service_name='Generalize_Regression' + _id_generator()
        output_name=output_service_name.replace(' ', '_')
    else:
        output_service_name=output_name.replace(' ', '_')

    output_service=_create_output_service(gis, output_name, output_service_name,
                                          'Generalized Linear Regression')

    params['output_name'] = _json.dumps({
        "serviceProperties": {"name" : output_name, "serviceUrl" : output_service.url},
        "itemProperties": {"itemId" : output_service.itemid}})


    _set_context(params)

    param_db={
        "input_layer": (_FeatureSet, "inputLayer"),
        #"output_name": (str, "outputName"),
        "features_to_predict" : (_FeatureSet, "featuresToPredict"),
        "dependent_variable" : (str, "dependentVariable"),
        "explanatory_variables" : (str, "explanatoryVariables"),
        "regression_family" : (str, "regressionFamily"),
        "generate_coef_table" : (bool, "generateCoefficientTable"),
        "variable_matching" : (list, "explanatoryVariableMatching"),
        "dependent_mapping" : (list, "dependentMapping"),
        #"context": (str, "context"),
        "output_name": (_FeatureSet, "outputName"),
        "context": (str, "context"),
        "output": (_FeatureSet, "Output Features"),
        #"output_name": (_FeatureSet, "OutputName")
    }
    #    "output_name": (_FeatureSet, "outputName"),
    #}


    #return_values=[
    #    {"name": "output", "display_name": "Output Features", "type": _FeatureSet},
    #]


    return_values=[
        {"name": "output", "display_name": "Output Features", "type": _FeatureSet},
    ]


    try:
        _execute_gp_tool(gis, "GeneralizedLinearRegression",
                         params, param_db,
                         return_values, _use_async,
                         url, True)
        return output_service
    except:
        output_service.delete()
        raise

    return

def find_point_clusters(
    input_layer,
    method,
    min_feature_clusters,
    search_distance=None,
    distance_unit=None,
    output_name=None,
    gis=None):
    """
    This tool extracts clusters from your input point features and identifies any surrounding noise.

    For example, a nongovernmental organization is studying a particular pest-borne disease. It has
    a point dataset representing households in a study area, some of which are infested, and some of
    which are not. By using the Find Point Clusters tool, an analyst can determine clusters of
    infested households to help pinpoint an area to begin treatment and extermination of pests.

    ==========================   ===============================================================
    **Argument**                 **Description**
    --------------------------   ---------------------------------------------------------------
    input_layer                  required FeatureSet, The table, point, line or polygon features
                                 containing potential incidents.
    --------------------------   ---------------------------------------------------------------
    method                       required String. The algorithm used for cluster analysis. This
                                 parameter must be specified as DBSCAN or HDBSCAN.
    --------------------------   ---------------------------------------------------------------
    min_feature_clusters         optional Integer. Minimum number of clusters to find in a dataset.
    --------------------------   ---------------------------------------------------------------
    search_distance              optional Float.  The distance to search between points to form
                                 a cluster.  This is required for DBSCAN.
    --------------------------   ---------------------------------------------------------------
    distance_unit                optional String. The `search_distance` units.
    --------------------------   ---------------------------------------------------------------
    output_name                  optional string, The task will create a feature service of the
                                 results. You define the name of the service.
    --------------------------   ---------------------------------------------------------------
    gis                          optional GIS, the GIS on which this tool runs. If not
                                 specified, the active GIS is used.
    ==========================   ===============================================================

    :returns:
       Output feature layer item

    """
    kwargs=locals()

    gis=_arcgis.env.active_gis if gis is None else gis
    url=gis.properties.helperServices.geoanalytics.url

    params={}
    for key, value in kwargs.items():
        if value is not None:
            params[key]=value

    if output_name is None:
        output_service_name='Find Point Clusters_' + _id_generator()
        output_name=output_service_name.replace(' ', '_')
    else:
        output_service_name=output_name.replace(' ', '_')

    output_service=_create_output_service(gis, output_name, output_service_name, 'Find Point Clusters')

    _set_context(params)

    param_db={
        "input_layer": (_FeatureSet, "inputLayer"),
        "method" : (str, "clusterMethod"),
        "min_feature_clusters": (int, "minFeaturesCluster"),
        "distance_unit": (str, "searchDistanceUnit"),
        "search_distance" : (float, "searchDistance"),
        "output_name": (str, "outputName"),
        "context": (str, "context"),
        "output": (_FeatureSet, "Output Features"),
    }
    return_values=[
        {"name": "output", "display_name": "Output Features", "type": _FeatureSet},
    ]

    try:
        _execute_gp_tool(gis, "FindPointClusters", params, param_db, return_values, _use_async, url, True)
        return output_service
    except:
        output_service.delete()
        raise

    return

def calculate_density(
    input_layer,
    fields=None,
    weight="""Uniform""",
    bin_type="""Square""",
    bin_size=None,
    bin_size_unit=None,
    time_step_interval=None,
    time_step_interval_unit=None,
    time_step_repeat_interval=None,
    time_step_repeat_interval_unit=None,
    time_step_reference=None,
    radius=None,
    radius_unit=None,
    area_units="""SquareKilometers""",
    output_name=None,
    gis=None):
    """




Parameters:

   input_layer: Input Points (Feature layer). Required parameter.

   fields: Population Field (str). Optional parameter.

   weight: Weight (str). Required parameter.
      Choice list:['Uniform', 'Kernel']

   bin_type: Output Bin Type (str). Required parameter.
      Choice list:['Square', 'Hexagon']

   bin_size: Output Bin Size (float). Required parameter.

   bin_size_unit: Output Bin Size Unit (str). Required parameter.
      Choice list:['Feet', 'Yards', 'Miles', 'Meters', 'Kilometers', 'NauticalMiles']

   time_step_interval: Time Step Interval (int). Optional parameter.

   time_step_interval_unit: Time Step Interval Unit (str). Optional parameter.
      Choice list:['Years', 'Months', 'Weeks', 'Days', 'Hours', 'Minutes', 'Seconds', 'Milliseconds']

   time_step_repeat_interval: Time Step Repeat Interval (int). Optional parameter.

   time_step_repeat_interval_unit: Time Step Repeat Interval Unit (str). Optional parameter.
      Choice list:['Years', 'Months', 'Weeks', 'Days', 'Hours', 'Minutes', 'Seconds', 'Milliseconds']

   time_step_reference: Time Step Reference (_datetime). Optional parameter.

   radius: Radius (float). Required parameter.

   radius_unit: Radius Unit (str). Required parameter.
      Choice list:['Feet', 'Yards', 'Miles', 'Meters', 'Kilometers', 'NauticalMiles']

   area_units: Area Unit Scale Factor (str). Optional parameter.
      Choice list:['SquareMeters', 'SquareKilometers', 'Hectares', 'SquareFeet', 'SquareYards', 'SquareMiles', 'Acres']

   output_name: Output Features Name (str). Required parameter.

   gis: Optional, the GIS on which this tool runs. If not specified, the active GIS is used.


Returns:
   output - Output Features as a feature layer collection item


    """
    kwargs=locals()

    gis=_arcgis.env.active_gis if gis is None else gis
    url=gis.properties.helperServices.geoanalytics.url

    params={}
    for key, value in kwargs.items():
        if value is not None:
            params[key]=value

    if output_name is None:
        output_service_name='Calculate Density Analysis_' + _id_generator()
        output_name=output_service_name.replace(' ', '_')
    else:
        output_service_name=output_name.replace(' ', '_')

    output_service=_create_output_service(gis, output_name, output_service_name, 'Calculate Density')

    params['output_name']=_json.dumps({
        "serviceProperties": {"name" : output_name, "serviceUrl" : output_service.url},
        "itemProperties": {"itemId" : output_service.itemid}})

    _set_context(params)

    param_db={
        "input_layer": (_FeatureSet, "inputLayer"),
        "fields": (str, "fields"),
        "weight": (str, "weight"),
        "bin_type": (str, "binType"),
        "bin_size": (float, "binSize"),
        "bin_size_unit": (str, "binSizeUnit"),
        "time_step_interval": (int, "timeStepInterval"),
        "time_step_interval_unit": (str, "timeStepIntervalUnit"),
        "time_step_repeat_interval": (int, "timeStepRepeatInterval"),
        "time_step_repeat_interval_unit": (str, "timeStepRepeatIntervalUnit"),
        "time_step_reference": (_datetime, "timeStepReference"),
        "radius": (float, "radius"),
        "radius_unit": (str, "radiusUnit"),
        "area_units": (str, "areaUnits"),
        "output_name": (str, "outputName"),
        "context": (str, "context"),
        "output": (_FeatureSet, "Output Features"),
    }
    return_values=[
        {"name": "output", "display_name": "Output Features", "type": _FeatureSet},
    ]

    try:
        _execute_gp_tool(gis, "CalculateDensity", params, param_db, return_values, _use_async, url, True)
        return output_service
    except:
        output_service.delete()
        raise


calculate_density.__annotations__={
    'fields': str,
    'weight': str,
    'bin_type': str,
    'bin_size': float,
    'bin_size_unit': str,
    'time_step_interval': int,
    'time_step_interval_unit': str,
    'time_step_repeat_interval': int,
    'time_step_repeat_interval_unit': str,
    'time_step_reference': _datetime,
    'radius': float,
    'radius_unit': str,
    'area_units': str,
    'output_name': str}


def find_hot_spots(
    point_layer,
    bin_size=5,
    bin_size_unit="Miles",
    neighborhood_distance=5,
    neighborhood_distance_unit="Miles",
    time_step_interval=None,
    time_step_interval_unit=None,
    time_step_alignment=None,
    time_step_reference=None,
    output_name=None,
    gis=None):
    """

    Parameters:

       point_layer: Input Points (FeatureSet). Required parameter.

       bin_size: Bin Size (float). Optional parameter.

       bin_size_unit: Bin Size Unit (str). Optional parameter.
          Choice list:['Feet', 'Yards', 'Miles', 'Meters', 'Kilometers', 'NauticalMiles']

       neighborhood_distance: Neighborhood Distance (float). Optional parameter.

       neighborhood_distance_unit: Neighborhood Distance Unit (str). Optional parameter.
          Choice list:['Feet', 'Yards', 'Miles', 'Meters', 'Kilometers', 'NauticalMiles']

       time_step_interval: Time Step Interval (int). Optional parameter.

       time_step_interval_unit: Time Step Interval Unit (str). Optional parameter.
          Choice list:['Years', 'Months', 'Weeks', 'Days', 'Hours', 'Minutes', 'Seconds', 'Milliseconds']

       time_step_alignment: Time Step Alignment (str). Optional parameter.
          Choice list:['EndTime', 'StartTime', 'ReferenceTime']

       time_step_reference: Time Step Reference (_datetime). Optional parameter.

       output_name: Output Features Name (str). Optional parameter.

       gis: Optional, the GIS on which this tool runs. If not specified, the active GIS is used.


    Returns:
       output - Output Features as a feature layer collection item


    """
    kwargs=locals()

    gis=_arcgis.env.active_gis if gis is None else gis
    url=gis.properties.helperServices.geoanalytics.url

    params={}
    for key, value in kwargs.items():
        if value is not None:
            params[key]=value

    if output_name is None:
        output_service_name='Hotspot Analysis_' + _id_generator()
        output_name=output_service_name.replace(' ', '_')
    else:
        output_service_name=output_name.replace(' ', '_')

    output_service=_create_output_service(gis, output_name, output_service_name, 'Find Hotspots')

    params['output_name']=_json.dumps({
        "serviceProperties": {"name" : output_name, "serviceUrl" : output_service.url},
        "itemProperties": {"itemId" : output_service.itemid}})

    _set_context(params)

    param_db={
        "point_layer": (_FeatureSet, "pointLayer"),
        "bin_size": (float, "binSize"),
        "bin_size_unit": (str, "binSizeUnit"),
        "neighborhood_distance": (float, "neighborhoodDistance"),
        "neighborhood_distance_unit": (str, "neighborhoodDistanceUnit"),
        "time_step_interval": (int, "timeStepInterval"),
        "time_step_interval_unit": (str, "timeStepIntervalUnit"),
        "time_step_alignment": (str, "timeStepAlignment"),
        "time_step_reference": (_datetime, "timeStepReference"),
        #"cell_size" : (int, "cellSize"),
        #"cell_size_units": (str, "cellSizeUnits"),
        #"shape_type" : (str, "shapeType"),
        "output_name": (str, "outputName"),
        "context": (str, "context"),
        "output": (_FeatureSet, "Output Features"),
    }
    return_values=[
        {"name": "output", "display_name": "Output Features", "type": _FeatureSet},
    ]

    try:
        _execute_gp_tool(gis, "FindHotSpots", params, param_db, return_values, _use_async, url, True)
        return output_service
    except:
        output_service.delete()
        raise

find_hot_spots.__annotations__={
    'bin_size': float,
    'bin_size_unit': str,
    'neighborhood_distance': float,
    'neighborhood_distance_unit': str,
    'time_step_interval': int,
    'time_step_interval_unit': str,
    'time_step_alignment': str,
    'time_step_reference': _datetime,
    'output_name': str}


def create_space_time_cube(point_layer: _FeatureSet,
                           bin_size: float,
                           bin_size_unit: str,
                           time_step_interval: int,
                           time_step_interval_unit: str,
                           time_step_alignment: str=None,
                           time_step_reference: _datetime=None,
                           summary_fields: str=None,
                           output_name: str=None,
                           context: str=None,
                           gis=None) -> DataFile:
    """
    Summarizes a set of points into a netCDF data structure by aggregating them into space-time bins. Within each bin,
    the points are counted and specified attributes are aggregated. For all bin locations, the trend for counts and
    summary field values are evaluated.

    Parameters:

       point_layer: Input Features (FeatureSet). Required parameter.

       bin_size: Distance Interval (float). Required parameter.

       bin_size_unit: Distance Interval Unit (str). Required parameter.
          Choice list:['Feet', 'Yards', 'Miles', 'Meters', 'Kilometers', 'NauticalMiles']

       time_step_interval: Time Step Interval (int). Required parameter.

       time_step_interval_unit: Time Step Interval Unit (str). Required parameter.
          Choice list:['Years', 'Months', 'Weeks', 'Days', 'Hours', 'Minutes', 'Seconds', 'Milliseconds']

       time_step_alignment: Time Step Alignment (str). Optional parameter.
          Choice list:['EndTime', 'StartTime', 'ReferenceTime']

       time_step_reference: Time Step Reference (datetime). Optional parameter.

       summary_fields: Summary Fields (str). Optional parameter.

       output_name: Output Name (str). Required parameter.

       context: Context (str). Optional parameter.

        gis: Optional, the GIS on which this tool runs. If not specified, the active GIS is used.


    Returns:
       output_cube - Output Space Time Cube as a DataFile

    """
    kwargs=locals()

    gis=_arcgis.env.active_gis if gis is None else gis
    url=gis.properties.helperServices.geoanalytics.url

    params={}
    for key, value in kwargs.items():
        if value is not None:
            params[key]=value

    _set_context(params)

    param_db={
        "point_layer": (_FeatureSet, "pointLayer"),
        "bin_size": (float, "binSize"),
        "bin_size_unit": (str, "binSizeUnit"),
        "time_step_interval": (int, "timeStepInterval"),
        "time_step_interval_unit": (str, "timeStepIntervalUnit"),
        "time_step_alignment": (str, "timeStepAlignment"),
        "time_step_reference": (_datetime, "timeStepReference"),
        "summary_fields": (str, "summaryFields"),
        "output_name": (str, "outputName"),
        "context": (str, "context"),
        "output_cube": (DataFile, "Output Space Time Cube"),
    }
    return_values=[
        {"name": "output_cube", "display_name": "Output Space Time Cube", "type": DataFile},
    ]

    return _execute_gp_tool(gis, "CreateSpaceTimeCube", params, param_db, return_values, _use_async, url, True)

