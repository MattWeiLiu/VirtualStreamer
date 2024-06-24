from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def vertexai_predict(project, location, endpoint_id, instances):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}

    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # The format of each instance should conform to the deployed model's prediction input schema.
    # instances = instances if type(instances) == list else [instances]
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    return response


def vertexai_predict_aiplatform(project, location, endpoint_id, instances):
    endpoint = aiplatform.Endpoint(
        endpoint_name="projects/{}/locations/{}/endpoints/{}".format(project, location, endpoint_id),
    )
    response = endpoint.predict(instances)
    return response
