import json
import os
import boto3
import logging
import uuid
import re
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Tuple, Union

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize boto3 clients - use resource optimization
bedrock_client = None
dynamodb_client = None
apigw_management_client = None

# Constants
SESSION_HISTORY_TABLE = os.environ.get('SESSION_HISTORY_TABLE', 'catalog_session_history')
CONNECTION_TABLE = os.environ.get('CONNECTION_TABLE', 'websocket_connections')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 10))
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 4096))
GENERAL_CONTEXT_TOKENS = int(os.environ.get('GENERAL_CONTEXT_TOKENS', 1000))
TIMEOUT_SECONDS = int(os.environ.get('TIMEOUT_SECONDS', 25))  # Lambda timeout buffer
CACHE_TTL = int(os.environ.get('CACHE_TTL', 300))  # Cache TTL in seconds
WORKFLOW_TIMEOUT = int(os.environ.get('WORKFLOW_TIMEOUT', 1800))  # 30 minutes workflow timeout

# Initialize thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Enhanced workflow schemas with state transition information
WORKFLOW_SCHEMAS = {
    "0": {   # General-chat
        "required_fields": [],
        "optional_fields": [
            "CONVERSATION"
        ],
        "states": ["INITIAL", "RESPONDING"],
        "transitions": {}
    },
    "1": {   # Clone-and-modify existing offer
        "required_fields": [
            "EXISTING_OFFER_NAME",
            "NEW_OFFER_NAME",
            "CATEGORY",
            "ENTITIES_TO_MODIFY",
            "CHANGES"
        ],
        "optional_fields": [
            "CONFIRMED",
            "CONVERSATION",
            "AVAILABLE_ENTITIES",
            "ACTION",
            "OVERRIDE_TYPES",
            "REFERENCE_ENTITY"
        ],
        "states": [
            "COLLECT_EXISTING_OFFER", 
            "COLLECT_NEW_NAME", 
            "COLLECT_CATEGORY", 
            "COLLECT_ENTITIES", 
            "COLLECT_CHANGES", 
            "CONFIRM_DETAILS",
            "COMPLETED"
        ],
        "transitions": {
            "COLLECT_EXISTING_OFFER": ["COLLECT_NEW_NAME"],
            "COLLECT_NEW_NAME": ["COLLECT_CATEGORY"],
            "COLLECT_CATEGORY": ["COLLECT_ENTITIES"],
            "COLLECT_ENTITIES": ["COLLECT_CHANGES"],
            "COLLECT_CHANGES": ["CONFIRM_DETAILS"],
            "CONFIRM_DETAILS": ["COMPLETED", "COLLECT_NEW_NAME", "COLLECT_CATEGORY", "COLLECT_ENTITIES", "COLLECT_CHANGES"],
            "COMPLETED": ["COLLECT_NEW_NAME", "COLLECT_CATEGORY", "COLLECT_ENTITIES", "COLLECT_CHANGES"]
        }
    },
    "2": {   # Create new entity
        "required_fields": [
            "ENTITY_TYPE",
            "ENTITY_SUBTYPE",
            "CATEGORY",
            "BASE_ON_EXISTING"
        ],
        "optional_fields": [
            "BROWSING_STATE",
            "CURRENT_PAGE",
            "NEW_NAME_NEEDED",
            "NEW_ENTITY_NAME",
            "REFERENCE_ENTITY",
            "AVAILABLE_ENTITIES",
            "ENTITIES_TO_MODIFY",
            "CHANGES",
            "CONFIGURATION_CHANGES",
            "ACTION",
            "OVERRIDE_TYPES",
            "CONVERSATION"
        ],
        "states": [
            "COLLECT_ENTITY_TYPE",
            "BROWSE_TEMPLATES",
            "COLLECT_ENTITY_SUBTYPE",
            "BROWSE_CATEGORIES",
            "COLLECT_CATEGORY",
            "COLLECT_BASE_PREFERENCE",
            "BROWSE_ENTITIES",
            "COLLECT_REFERENCE_ENTITY",
            "COLLECT_ENTITIES_TO_MODIFY",
            "COLLECT_CHANGES",
            "COLLECT_NAME",
            "COLLECT_CONFIG",
            "CONFIRM_DETAILS",
            "COMPLETED"
        ],
        "transitions": {
            # Transitions defined here, simplified for brevity
        }
    }
}

classifier_prompt = """
You are an AI assistant that classifies telecom product catalog queries into exactly one of three workflows:
  • 0 = General (any question not about creating or modifying telecom offers)  
  • 1 = Clone-and-modify existing offer (user wants to create a new offer based on an existing one)
  • 2 = Create new entity from scratch (user wants to build a brand‑new entity without referencing existing ones)

WORKFLOW CLASSIFICATION RULES:
 1) CLONE-AND-MODIFY (1): Assign when the user wants to create a new offer while referencing an existing one.
    - Look for phrases like "similar to", "based on", "like the", "clone", etc.
    - Key indicator: User mentions BOTH creating something new AND references an existing offer.
    
 2) CREATE NEW (2): Assign when the user wants to create a brand new entity without reference to existing ones.
    - Look for phrases like "create a new", "make a fresh", "add a", etc., without referencing existing entities.
    - Key indicator: User mentions creating an entity but does NOT reference any existing offers.
    
 0) GENERAL (0): Assign for all other queries.
    - Simple questions about the catalog
    - Browsing requests
    - Modifying an existing offer without creating a new one
    - Any conversation not clearly about creating new catalog entities

Return ONLY the digit 0, 1, or 2—no extra text.

User‑Query:
{user_query}

Workflow classification:
"""

clone_modify_prompt = """
You are a Catalog Assistant helping telecom product managers create new offers based on existing ones. Extract these JSON fields from the user input:

{{
  "EXISTING_OFFER_NAME": "Name of existing offer to use as reference (e.g., Gold5000, all4000)",
  "NEW_OFFER_NAME": "Name for new offer being created (e.g., Gold6000, all5000)",
  "CATEGORY": "Category for new offer (e.g., WhatsAppSim, Mobile Offer, PrepaidMobile)",
  "ENTITIES_TO_MODIFY": "Components that need modification (e.g., MSISDN, Recurring Charge, Data Allowance)",
  "CHANGES": "Specific changes for each component (e.g., limit to 2 MSISDNs, increase charge by 10%)"
}}

IMPORTANT GUIDELINES:
1. Set any missing values to "NONE" - never invent information
2. Extract exact entity names/values directly from the user's text
3. For CHANGES, capture the exact modification request as stated
4. If multiple changes are mentioned, include them all in the CHANGES field, separated by semicolons

Respond with ONLY the JSON object.

User says: {user_input}
"""

create_entity_prompt = """
You are a Catalog Assistant helping telecom product managers create new entities. Extract these JSON fields from the user input:

{{
  "ENTITY_TYPE": "Type of entity (Package, Bundle, Promotion, Component, Component Group)",
  "ENTITY_SUBTYPE": "Subtype or template (e.g., Mobile Base Offer, Fixed Line Package)",
  "CATEGORY": "Category for the entity (e.g., Mobile Offer, WhatsAppSim, PrepaidMobile)",
  "BASE_ON_EXISTING": "Whether to base on existing entity (YES, NO, or BROWSE)",
  "CONFIGURATION_CHANGES": "Any specific configuration details mentioned"
}}

IMPORTANT GUIDELINES:
1. Set any missing values to "NONE" - never invent information
2. For ENTITY_TYPE, select the closest match from the available types
3. For BASE_ON_EXISTING use:
   - "YES" if user specifically wants to base on an existing entity
   - "NO" if user specifically wants to create from scratch
   - "BROWSE" if user wants to see existing entities or is unclear
   - "NONE" if not specified

Respond with ONLY the JSON object.

User says: {user_input}
"""

followup_prompt = """
You are a conversational telecom product catalog assistant. Based on the partially completed workflow data and what's still missing, create ONE natural follow-up question.

Workflow Type: {workflow_type}
{schema_info}

Current information:
{partial_json}

GUIDELINES FOR YOUR QUESTION:
1. Make it conversational and friendly, not robotic
2. Reference information already collected for context
3. Ask about ONLY the next logical piece of information needed
4. For entity selection, offer to show options if appropriate
5. If asking about modifications, be specific about what types of changes can be made

Your single follow-up question:
"""

SCHEMA_INFO = {
    "1": """
This is a Clone-and-Modify workflow where the user wants to create a new offer based on an existing one.

Required fields:
- EXISTING_OFFER_NAME: The reference offer to base the new one on (e.g., Gold5000, all4000)
- NEW_OFFER_NAME: The name for the new offer being created
- CATEGORY: The category to place the new offer in (e.g., WhatsAppSim, Mobile Offer)
- ENTITIES_TO_MODIFY: Which components of the offer need modification (e.g., MSISDN, Recurring Charge)
- CHANGES: Specific changes to make to those components (e.g., limit to 2 MSISDNs, increase by 10%)

Optional fields that may be present:
- CONFIRMED: Whether user has confirmed the details
- AVAILABLE_ENTITIES: List of modifiable entities in the reference offer
- ACTION: Type of action (add/remove/modify/clone)
- OVERRIDE_TYPES: Classification of what kinds of changes are being made
""",

    "2": """
This is a Create New Entity workflow where the user wants to create a brand new telecom catalog entity.

Required fields:
- ENTITY_TYPE: Type of entity (Package, Bundle, Promotion, Component, Component Group)
- ENTITY_SUBTYPE: The template or subtype to use (e.g., Mobile Base Offer, Fixed Line Package)
- CATEGORY: Category to place the new entity in (e.g., Mobile Offer, WhatsAppSim)
- BASE_ON_EXISTING: Whether to base on existing entity (YES/NO/BROWSE)

Optional fields that may be present:
- BROWSING_STATE: Current browsing context if user is selecting from options
- NEW_ENTITY_NAME: Name for the new entity being created
- REFERENCE_ENTITY: Name of reference entity if basing on existing
- AVAILABLE_ENTITIES: List of modifiable components in the reference entity
- ENTITIES_TO_MODIFY: Components the user wants to change
- CHANGES: Specific changes to make to those components
- CONFIGURATION_CHANGES: Configuration details for from-scratch creation
"""
}

# Cache for connections and sessions to reduce DynamoDB calls
_connection_cache = {}
_session_cache = {}

# Enhanced conversation detection patterns
CONVERSATIONAL_PATTERNS = [
    r"^(hello|hi|hey|greetings)[\s\,\.\!]*$",
    r"^how are you",
    r"^good (morning|afternoon|evening)",
    r"^what('s| is) up",
    r"^(can|could) you help me",
]

EXIT_WORKFLOW_PATTERNS = [
    r"\b(exit|quit|done|cancel|stop|end)\b.*\b(workflow|process|operation)\b",
    r"\b(go|switch)\b.*\b(back|general|different)\b",
    r"\b(new|different|another)\b.*\b(topic|question|subject)\b",
    r"\bnever mind\b",
    r"\blet's talk about something else\b",
    r"^(start|begin) (over|again|fresh)",
]


def lambda_handler(event, context):
    """
    Main handler for WebSocket API Gateway events.
    Implements routing logic for different WebSocket event types.
    """
    try:
        print(f"Event received: {json.dumps(event)}")
        
        # Extract connection ID and route key
        request_context = event.get('requestContext', {})
        connection_id = request_context.get('connectionId')
        route_key = request_context.get('routeKey')
        
        print(f"Route key: {route_key}, Connection ID: {connection_id}")
        
        # Get domain and stage for API Gateway client
        domain_name = request_context.get('domainName')
        stage = request_context.get('stage')
        
        # Calculate remaining execution time for timeout management
        remaining_time = context.get_remaining_time_in_millis() if hasattr(context, 'get_remaining_time_in_millis') else 60000
        
        # Handle different WebSocket events
        if route_key == '$connect':
            return handle_connect(connection_id)
        elif route_key == '$disconnect':
            return handle_disconnect(connection_id)
        elif route_key == 'process_request':
            return handle_process_request(event, connection_id, domain_name, stage, remaining_time)
        else:
            print(f"Unknown route key: {route_key}")
            return {'statusCode': 400, 'body': 'Unknown route'}
            
    except Exception as e:
        print(f"Unhandled exception in lambda_handler: {str(e)}")
        traceback.print_exc()
        return {'statusCode': 500, 'body': 'Internal server error'}


def handle_connect(connection_id):
    """
    Handle new WebSocket connections.
    Stores connection information in DynamoDB and cache.
    """
    print(f"New connection: {connection_id}")
    try:
        # Store connection ID in DynamoDB for tracking active connections
        timestamp = datetime.now().isoformat()
        
        get_dynamodb_client().put_item(
            TableName=CONNECTION_TABLE,
            Item={
                'connectionId': {'S': connection_id},
                'timestamp': {'S': timestamp},
                'status': {'S': 'CONNECTED'}
            }
        )
        
        # Update connection cache
        _connection_cache[connection_id] = {
            'timestamp': timestamp,
            'status': 'CONNECTED'
        }
        
        return {'statusCode': 200, 'body': 'Connected'}
    except Exception as e:
        print(f"Error in connect handler: {str(e)}")
        return {'statusCode': 500, 'body': 'Connection error'}


def handle_disconnect(connection_id):
    """
    Handle WebSocket disconnections.
    Removes connection from DynamoDB and cache.
    """
    print(f"Disconnection: {connection_id}")
    try:
        # Remove connection ID from DynamoDB
        get_dynamodb_client().delete_item(
            TableName=CONNECTION_TABLE,
            Key={'connectionId': {'S': connection_id}}
        )
        
        # Remove from connection cache if exists
        if connection_id in _connection_cache:
            del _connection_cache[connection_id]
        
        return {'statusCode': 200, 'body': 'Disconnected'}
    except Exception as e:
        print(f"Error in disconnect handler: {str(e)}")
        return {'statusCode': 500, 'body': 'Disconnection error'}


def handle_process_request(event, connection_id, domain_name, stage, remaining_time_ms):
    """
    Handle processing requests from the client—synchronously.
    """
    try:
        if 'body' not in event:
            raise ValueError("No message body transmitted")

        body = json.loads(event['body'])
        action = body.get('action')
        if action != 'process_request':
            raise ValueError(f"Unknown action: {action}")

        customer_id = body.get('customerId', 'unknown')
        query = body.get('query', '').strip()
        session_id = body.get('sessionId') or str(uuid.uuid4())

        if not query:
            send_response(connection_id, domain_name, stage, {
                'type': 'error',
                'message': 'Query cannot be empty'
            })
            return {'statusCode': 400, 'body': 'Empty query'}

        # Compute timeout buffer for process_query_and_respond
        timeout_buffer_ms = min(remaining_time_ms - 5000, TIMEOUT_SECONDS * 1000)

        # Synchronous call to process query and respond
        process_query_and_respond(
            query,
            session_id,
            connection_id,
            domain_name,
            stage,
            customer_id,
            timeout_buffer_ms
        )

        return {'statusCode': 200, 'body': ''}

    except Exception as e:
        print(f"Error in handle_process_request: {e}", traceback.format_exc())
        send_error(connection_id, domain_name, stage, str(e))
        return {'statusCode': 500, 'body': 'Error processing message'}

def classify_query(query: str) -> str:
    """
    Classifies a user query into the appropriate workflow type.
    Returns the workflow ID as a string.
    """
    # Clean the query
    query_lower = query.lower().strip()
    
    # Check for clone-and-modify patterns
    clone_patterns = [
        r"(?:clone|copy|duplicate|based on|similar to).*(?:offer|package|bundle)",
        r"(?:create|make).*(?:new|another).*(?:offer|package|bundle).*(?:like|based on|from)",
        r"(?:modify|change|update|adjust).*(?:existing|current)",
        r"(?:new version of).*(?:offer|package|bundle)",
    ]
    
    for pattern in clone_patterns:
        if re.search(pattern, query_lower):
            return "1"  # Clone-and-modify workflow
    
    # Check for create entity patterns
    create_patterns = [
        r"(?:create|make|add).*(?:new|another).*(?:package|bundle|component|promotion|entity)",
        r"(?:how|can i).*(?:create|make|add).*(?:new|another)",
        r"(?:create entity|make entity|new entity)",
        r"(?:create from scratch)",
    ]
    
    for pattern in create_patterns:
        if re.search(pattern, query_lower):
            return "2"  # Create entity workflow
    
    # Default to general chat
    return "0"


def process_query_and_respond(
    query: str,
    session_id: str,
    connection_id: str,
    domain_name: str,
    stage: str,
    customer_id: str,
    timeout_ms: int
) -> None:
    """
    Enhanced process_query_and_respond with robust state management and error recovery.
    """
    start_time = datetime.now()

    # 1) Load or init context
    session_context = get_session_context(session_id)
    print(f"Session context loaded: {json.dumps(session_context, default=str)}")
    
    # Check for session timeout
    if should_reset_workflow_timeout(session_context):
        print("Workflow timeout detected - resetting workflow state")
        session_context['current_workflow'] = "0"
        session_context['workflow_data'] = {}
        if 'workflow_completed' in session_context:
            session_context.pop('workflow_completed')
        if 'workflow_state' in session_context:
            session_context.pop('workflow_state')
    
    # 2) Determine workflow based on context
    current_wf = session_context.get('current_workflow', '0')
    
    # Check for conversational reset
    if is_conversational_input(query) and current_wf != "0":
        print("Detected conversational reset")
        current_wf = "0"
        session_context['current_workflow'] = "0"
        session_context['workflow_data'] = {}
        if 'workflow_completed' in session_context:
            session_context.pop('workflow_completed')
        if 'workflow_state' in session_context:
            session_context.pop('workflow_state')
    
    # Check if user is explicitly trying to exit the current workflow
    if should_exit_workflow(query) and current_wf != "0":
        print("User requested to exit current workflow")
        current_wf = "0"
        session_context['current_workflow'] = "0"
        session_context['workflow_data'] = {}
        if 'workflow_completed' in session_context:
            session_context.pop('workflow_completed')
        if 'workflow_state' in session_context:
            session_context.pop('workflow_state')
        
        exit_message = (
            "I've reset our current conversation. What would you like to do now? "
            "I can help you create new offers, modify existing ones, or answer general questions about the catalog."
        )
        
        payload = {
            'type': 'query_response',
            'sessionId': session_id,
            'response': exit_message,
            'workflow': "0",
            'timestamp': datetime.now().isoformat()
        }
        
        send_response(connection_id, domain_name, stage, payload)
        
        # Update session history
        update_session_history(
            session_id, customer_id, query, "0", 
            exit_message, {}
        )
        return
    
    # Handle specific post-completion actions for workflows
    if current_wf != "0" and session_context.get('workflow_completed', False):
        # Process post-completion scenarios
        result = process_post_completion_action(
            query, current_wf, session_context,
            session_id, connection_id, domain_name, stage, customer_id
        )
        
        if result:
            # If handled, just return
            return
    
    # Only classify if we're not already in a workflow or if workflow was just reset
    if current_wf == '0':
        workflow = classify_query(query)
        # Update the current workflow in session
        session_context['current_workflow'] = workflow
        
        # Initialize workflow state if needed
        if workflow != "0":
            if workflow == "1":  # Clone-modify workflow
                session_context['workflow_state'] = "COLLECT_EXISTING_OFFER"
            elif workflow == "2":  # Create entity workflow
                session_context['workflow_state'] = "COLLECT_ENTITY_TYPE"
    else:
        # We're already in a workflow, continue with it
        workflow = current_wf
    
    print(f"Using workflow: {workflow}, State: {session_context.get('workflow_state', 'N/A')}")

    # 3) Process based on workflow
    if workflow == "0":
        # General chat
        response_text = process_general_query(query, session_context)
        workflow_data = {}

        # Send response
        payload = {
            'type': 'query_response',
            'sessionId': session_id,
            'response': response_text,
            'workflow': workflow,
            'timestamp': datetime.now().isoformat()
        }
        send_response(connection_id, domain_name, stage, payload)
    else:
        # Catalog workflow - handle based on type
        try:
            if workflow == "1":
                json_response, workflow_data = process_clone_modify_offer(query, session_context)
            elif workflow == "2":
                json_response, workflow_data = process_create_entity(query, session_context)
            else:
                json_response, workflow_data = "Error: Invalid workflow", {}
            
            # Store workflow data in session context
            session_context['workflow_data'] = workflow_data
            
            # Check if the workflow is complete
            if is_workflow_data_complete(workflow, workflow_data):
                # All required fields are filled, send the final response
                final_message = generate_final_response(workflow, workflow_data)
                
                payload = {
                    'type': 'query_response',
                    'sessionId': session_id,
                    'response': final_message,
                    'workflow': workflow,
                    'workflow_data': workflow_data,
                    'is_complete': True,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Set workflow completion flag
                session_context['workflow_completed'] = True
                
                send_response(connection_id, domain_name, stage, payload)
            else:
                # Still missing fields, send follow-up question
                followup = generate_followup_question(workflow, workflow_data)
                
                payload = {
                    'type': 'follow_up',
                    'sessionId': session_id,
                    'message': followup,
                    'workflow': workflow,
                    'workflow_data': workflow_data,
                    'is_complete': False
                }
                send_response(connection_id, domain_name, stage, payload)
        except Exception as e:
            print(f"Error processing workflow {workflow}: {str(e)}")
            traceback.print_exc()
            
            # Send error message and fallback to general chat
            error_message = (
                "I encountered an issue processing your request. Let's start over. "
                "Could you please rephrase what you'd like to do?"
            )
            
            payload = {
                'type': 'error',
                'sessionId': session_id,
                'message': error_message,
                'workflow': "0"  # Reset to general workflow
            }
            
            send_response(connection_id, domain_name, stage, payload)
            
            # Reset workflow
            session_context['current_workflow'] = "0"
            session_context['workflow_data'] = {}
            if 'workflow_completed' in session_context:
                session_context.pop('workflow_completed')
            if 'workflow_state' in session_context:
                session_context.pop('workflow_state')

    # 4) Update session history
    update_session_history(
        session_id,
        customer_id,
        query,
        workflow,
        workflow_data.get('CONVERSATION', '') if workflow != "0" else response_text,
        workflow_data if workflow != "0" else {}
    )
    
    print(f"Processing completed in {(datetime.now() - start_time).total_seconds()}s")


def is_conversational_input(query: str) -> bool:
    """
    Detect if user input is a conversational greeting or small talk.
    """
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.search(pattern, query.lower()):
            return True
    return False


def should_exit_workflow(query: str) -> bool:
    """
    Detect if user is trying to exit the current workflow.
    """
    for pattern in EXIT_WORKFLOW_PATTERNS:
        if re.search(pattern, query.lower()):
            return True
    return False


def should_reset_workflow_timeout(session_context: Dict) -> bool:
    """
    Check if the current workflow has timed out and should be reset.
    """
    # If no workflow or already in general chat, no timeout needed
    if session_context.get('current_workflow', '0') == '0':
        return False
    
    # Check last update timestamp
    last_updated = session_context.get('last_updated')
    if not last_updated:
        return False
    
    try:
        last_update_time = datetime.fromisoformat(last_updated)
        # If it's been more than WORKFLOW_TIMEOUT seconds, reset
        if (datetime.now() - last_update_time).total_seconds() > WORKFLOW_TIMEOUT:
            return True
    except (ValueError, TypeError):
        # If we can't parse the timestamp, don't reset
        return False
    
    return False


def process_post_completion_action(
    query: str,
    current_wf: str,
    session_context: Dict,
    session_id: str,
    connection_id: str,
    domain_name: str,
    stage: str,
    customer_id: str
) -> bool:
    """
    Process user actions after a workflow has been completed.
    Returns True if the action was handled, False otherwise.
    """
    # Check if user wants to make additional changes to a completed offer
    if re.search(r"\b(additional|more|further)\s+changes\b", query, re.IGNORECASE) or \
       re.search(r"\b(modify|change|update|edit)\b", query, re.IGNORECASE) or \
       re.search(r"\bmake\s+.*\s+changes\b", query, re.IGNORECASE) or \
       re.search(r"^(1|option 1)", query, re.IGNORECASE):
        
        print("User requested additional changes to completed workflow")
        
        # Reset completion flag but keep the workflow data
        session_context['workflow_completed'] = False
        
        workflow_data = session_context.get('workflow_data', {})
        
        # Handle based on workflow type
        if current_wf == "1":  # Clone-modify workflow
            # Ask which aspect they want to modify
            ask_modify_message = generate_modification_options_message(workflow_data, "1")
            
            # Update the conversation field and reset confirmation
            workflow_data['CONVERSATION'] = ask_modify_message
            if 'CONFIRMED' in workflow_data:
                workflow_data.pop('CONFIRMED')
            
            # Send follow-up question
            payload = {
                'type': 'follow_up',
                'sessionId': session_id,
                'message': ask_modify_message,
                'workflow': current_wf,
                'workflow_data': workflow_data,
                'is_complete': False
            }
            
            send_response(connection_id, domain_name, stage, payload)
            
            # Update session context
            session_context['workflow_data'] = workflow_data
            session_context['workflow_state'] = 'CONFIRM_DETAILS'
            
            # Update session history
            update_session_history(
                session_id, customer_id, query, current_wf, 
                ask_modify_message, workflow_data
            )
            return True
            
        elif current_wf == "2":  # Create entity workflow
            # Ask which aspect they want to modify
            ask_modify_message = generate_modification_options_message(workflow_data, "2")
            
            # Update the conversation field and reset confirmation
            workflow_data['CONVERSATION'] = ask_modify_message
            if 'CONFIRMED' in workflow_data:
                workflow_data.pop('CONFIRMED')
            
            # Send follow-up question
            payload = {
                'type': 'follow_up',
                'sessionId': session_id,
                'message': ask_modify_message,
                'workflow': current_wf,
                'workflow_data': workflow_data,
                'is_complete': False
            }
            
            send_response(connection_id, domain_name, stage, payload)
            
            # Update session context
            session_context['workflow_data'] = workflow_data
            session_context['workflow_state'] = 'CONFIRM_DETAILS'
            
            # Update session history
            update_session_history(
                session_id, customer_id, query, current_wf, 
                ask_modify_message, workflow_data
            )
            return True
    
    # Check if user wants to create another entity/offer
    elif re.search(r"\b(create another|new|another)\b", query, re.IGNORECASE) or \
         re.search(r"^(2|option 2)", query, re.IGNORECASE):
        
        print("User requested to create another entity")
        
        # Reset workflow to start fresh
        session_context['current_workflow'] = "0"  # Will be reclassified
        session_context['workflow_data'] = {}
        session_context.pop('workflow_completed', None)
        session_context.pop('workflow_state', None)
        
        # Send confirmation message
        confirmation = "I'll help you create a new entity. What type of entity would you like to create?"
        
        payload = {
            'type': 'query_response',
            'sessionId': session_id,
            'response': confirmation,
            'workflow': "0",  # Reset to default for reclassification
            'timestamp': datetime.now().isoformat()
        }
        
        send_response(connection_id, domain_name, stage, payload)
        
        # Update session history
        update_session_history(
            session_id, customer_id, query, "0", 
            confirmation, {}
        )
        return True
    
    # Check if user wants to proceed to deployment
    elif re.search(r"\b(proceed|deploy|deployment)\b", query, re.IGNORECASE) or \
         re.search(r"^(3|option 3)", query, re.IGNORECASE):
        
        print("User requested to proceed to deployment")
        
        # Handle deployment confirmation
        workflow_data = session_context.get('workflow_data', {})
        
        if current_wf == "1":
            entity_name = workflow_data.get('NEW_OFFER_NAME', 'new offer')
        else:
            entity_type = workflow_data.get('ENTITY_TYPE', 'entity')
            entity_name = workflow_data.get('NEW_ENTITY_NAME', f'new {entity_type}')
        
        deploy_message = (
            f"Great! The **{entity_name}** "
            f"has been queued for deployment to the catalog system. "
            f"The deployment process typically takes 5-10 minutes to complete. "
            f"You'll receive a notification once the deployment is finished.\n\n"
            f"Is there anything else you would like help with?"
        )
        
        # Send response
        payload = {
            'type': 'query_response',
            'sessionId': session_id,
            'response': deploy_message,
            'workflow': "0",  # Reset to general workflow
            'timestamp': datetime.now().isoformat()
        }
        
        send_response(connection_id, domain_name, stage, payload)
        
        # Reset workflow
        session_context['current_workflow'] = "0"
        session_context['workflow_data'] = {}
        session_context.pop('workflow_completed', None)
        session_context.pop('workflow_state', None)
        
        # Update session history
        update_session_history(
            session_id, customer_id, query, "0", 
            deploy_message, {}
        )
        return True
    
    # Not a post-completion action
    return False


def generate_modification_options_message(workflow_data: Dict, workflow_type: str) -> str:
    """
    Generate a message with options for modifying a completed entity.
    """
    if workflow_type == "1":  # Clone-modify workflow
        return (
            f"What would you like to change about the **{workflow_data.get('NEW_OFFER_NAME', 'new offer')}**?\n\n"
            f"You can modify:\n"
            f"• The offer name\n"
            f"• The category\n"
            f"• Components to modify\n"
            f"• The specific changes to those components"
        )
    else:  # Create entity workflow
        entity_type = workflow_data.get('ENTITY_TYPE', 'entity')
        entity_name = workflow_data.get('NEW_ENTITY_NAME', f'new {entity_type}')
        
        return (
            f"What would you like to change about the **{entity_name}**?\n\n"
            f"You can modify:\n"
            f"• The {entity_type} name\n"
            f"• The category\n"
            f"• The subtype or template\n"
            f"• The configuration details"
        )

def get_entity_types_and_templates(page: int = 1) -> str:
    """
    Returns available entity types and templates for browsing.
    """
    # Mock implementation - would connect to actual catalog in production
    entity_templates = {
        "Package": ["Mobile Base Offer", "Fixed Line Package", "Data-only Package", "Family Package", "Business Package"],
        "Bundle": ["Entertainment Bundle", "Security Bundle", "Productivity Bundle"],
        "Component": ["SIM Component", "Usage Component", "Charging Component"],
        "Promotion": ["Seasonal Promotion", "Loyalty Promotion", "Acquisition Promotion"],
        "Component Group": ["Base Components", "Add-on Components", "Charging Components"]
    }
    
    items_per_page = 5
    
    # Calculate total items and pages
    total_templates = 0
    entity_types_list = []
    
    for entity_type, templates in entity_templates.items():
        total_templates += len(templates)
        entity_types_list.append(entity_type)
    
    total_pages = (total_templates + items_per_page - 1) // items_per_page
    
    # Ensure page is within bounds
    page = max(1, min(page, total_pages))
    
    response = "**Available Entity Types and Templates:**\n\n"
    
    current_item = 0
    items_on_current_page = 0
    
    for entity_type, templates in entity_templates.items():
        # Skip items before the current page
        if current_item + len(templates) <= (page - 1) * items_per_page:
            current_item += len(templates)
            continue
            
        # Start displaying this entity type
        response += f"**{entity_type}**\n"
        
        for i, template in enumerate(templates, 1):
            # Skip items before the current page
            if current_item < (page - 1) * items_per_page:
                current_item += 1
                continue
                
            response += f"{i}. {template}\n"
            current_item += 1
            items_on_current_page += 1
            
            # Stop if we've filled the page
            if items_on_current_page >= items_per_page:
                break
                
        response += "\n"
        
        # Stop if we've filled the page
        if items_on_current_page >= items_per_page:
            break
    
    # Add pagination information
    response += f"\n*Showing page {page} of {total_pages}*"
    
    if page < total_pages:
        response += " - Type 'more' to see additional options"
    
    response += "\n\nSelect a template by typing its number or name (e.g., 'Select Mobile Base Offer')"
    response += "\nOr type 'exit browsing' to choose a different approach"
    
    return response


def get_category_list(entity_type: str, template: str, page: int = 1) -> str:
    """
    Returns available categories for a given entity type and template.
    """
    # Mock category data - would come from actual catalog in production
    category_lists = {
        "Mobile Base Offer": ["Mobile Offer", "WhatsAppSim", "PrepaidMobile", "PostpaidMobile", "Corporate"],
        "Fixed Line Package": ["Home Internet", "Business Internet", "IPTV", "VoIP"],
        "Data-only Package": ["MobileBroadband", "IoT", "M2M", "TabletData"],
        "Entertainment Bundle": ["StreamingTV", "GamingBundle", "MusicStreaming"],
        "Security Bundle": ["DeviceSecurity", "FamilySafety", "DataProtection"]
    }
    
    # Get categories for the selected template
    categories = category_lists.get(template, [])
    
    # If no categories found, provide helpful message
    if not categories:
        return (
            f"**No predefined categories found for {template}**\n\n"
            f"Would you like to:\n"
            f"1. Create a new category\n"
            f"2. Select a different template\n"
            f"3. Browse all available categories\n\n"
            f"Please type your choice (1-3) or enter a custom category name."
        )
    
    items_per_page = 5
    total_pages = (len(categories) + items_per_page - 1) // items_per_page
    
    # Ensure page is within bounds
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(categories))
    
    visible_categories = categories[start_idx:end_idx]
    
    response = f"**Available Categories for {template}:**\n\n"
    for i, category in enumerate(visible_categories, 1):
        response += f"{i}. {category}\n"
    
    response += f"\n*Showing {len(visible_categories)} of {len(categories)} categories - Page {page} of {total_pages}*"
    
    if page < total_pages:
        response += "\nType 'more' to see additional categories"
    
    response += "\n\nSelect a category by typing its number or name (e.g., 'Select Mobile Offer')"
    response += "\nOr type a new category name to create a custom category"
    response += "\nOr type 'exit browsing' to choose a different approach"
    
    return response


def get_entity_list_by_template_category(entity_type: str, template: str, category: str, page: int = 1) -> str:
    """
    Returns available entities filtered by template and category.
    """
    # Mock entity data - would come from actual catalog in production
    entity_lists = {
        "Mobile Base Offer:Mobile Offer": ["Gold5000", "Silver3000", "Bronze1000", "Premium8000", "Basic500"],
        "Mobile Base Offer:WhatsAppSim": ["all4000", "all2000", "all1000", "plus5000", "lite500"],
        "Fixed Line Package:Home Internet": ["HomeFiber100", "HomeFiber500", "HomeWireless50", "HomePlus200"],
        "Entertainment Bundle:StreamingTV": ["PremiumTV", "BasicTV", "UltimateTV"],
        "Security Bundle:FamilySafety": ["ParentalControl", "LocationTracker", "ScreenTime"]
    }
    
    key = f"{template}:{category}"
    entities = entity_lists.get(key, [])
    
    # If no entities found, provide helpful message
    if not entities:
        return (
            f"**No existing {entity_type}s found in {category} category**\n\n"
            f"Would you like to:\n"
            f"1. Create a brand new {entity_type} from scratch\n"
            f"2. Browse a different category\n"
            f"3. Select a different template\n\n"
            f"Please type your choice (1-3) or type 'exit browsing' to start over."
        )
    
    items_per_page = 5
    total_pages = (len(entities) + items_per_page - 1) // items_per_page
    
    # Ensure page is within bounds
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(entities))
    
    visible_entities = entities[start_idx:end_idx]
    
    response = f"**Available {entity_type}s in {category}:**\n\n"
    for i, entity in enumerate(visible_entities, 1):
        response += f"{i}. {entity}\n"
    
    response += f"\n*Showing {len(visible_entities)} of {len(entities)} {entity_type.lower()}s - Page {page} of {total_pages}*"
    
    if page < total_pages:
        response += "\nType 'more' to see additional options"
    
    response += "\n\nSelect an entity by typing its number or name (e.g., 'Select Gold5000')"
    response += "\nOr type 'create new' to make a brand new entity without using these as reference"
    response += "\nOr type 'exit browsing' to choose a different approach"
    
    return response


def get_entity_by_selection(selection: str, entity_type: str, available_entities: List[str] = None) -> str:
    """
    Resolves a user selection (by number or name) to an actual entity.
    Enhanced with validation against available entities.
    """
    # If available entities provided, use those
    if available_entities:
        entity_list = available_entities
    else:
        # Mock entity data for demonstration
        all_entities = {
            "Package": ["Gold5000", "Silver3000", "all4000", "Premium8000", "Basic500"],
            "Bundle": ["StreamingBundle", "SecurityBundle", "ProductivityBundle"],
            "Component": ["MSISDN", "Recurring Charge", "Data Allowance"],
            "Promotion": ["SummerPromo", "NewCustomerDiscount", "LoyaltyBonus"],
            "Component Group": ["BaseComponents", "ChargingComponents", "UsageComponents"],
            "Template": ["Mobile Base Offer", "Fixed Line Package", "Data-only Package"],
            "Category": ["Mobile Offer", "WhatsAppSim", "HomeInternet"]
        }
        
        entity_list = all_entities.get(entity_type, [])
    
    # If selection is a number, treat as an index
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(entity_list):
            return entity_list[idx]
    
    # Otherwise, try to match by name (case insensitive partial match)
    selection_lower = selection.lower()
    for entity in entity_list:
        if selection_lower in entity.lower():
            return entity
        
    # If exact match not found, try fuzzy matching
    best_match = None
    highest_similarity = 0
    
    for entity in entity_list:
        # Simple similarity score based on common substring
        similarity = calculate_similarity(selection_lower, entity.lower())
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = entity
    
    # If we found a reasonably good match
    if highest_similarity > 0.6 and best_match:
        return best_match
    
    # If no match found, return the selection as-is
    return selection


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate simple string similarity for fuzzy matching.
    Returns a score between 0 and 1 where 1 is perfect match.
    """
    # Count common characters
    common_chars = sum(1 for c in str1 if c in str2)
    
    # Normalize by length of shorter string
    return common_chars / max(1, min(len(str1), len(str2)))


def get_available_entities(reference_entity: str) -> List[str]:
    """
    Returns list of modifiable entities in a reference entity.
    This is a mock implementation with error handling.
    """
    # Mock data - would come from actual catalog in production
    entity_components = {
        "Gold5000": ["MSISDN", "Recurring Charge", "Validity Period", "Data Allowance", "Voice Minutes", "SMS Count"],
        "all4000": ["MSISDN", "Recurring Charge", "Data Allowance", "WhatsApp Access", "Validity Period"],
        "Premium8000": ["MSISDN", "Recurring Charge", "Premium Content", "Data Allowance", "International Roaming"],
        "Silver3000": ["MSISDN", "Recurring Charge", "Data Allowance", "Voice Minutes"],
        "Bronze1000": ["MSISDN", "Recurring Charge", "Data Allowance"],
        "all2000": ["MSISDN", "Recurring Charge", "Data Allowance", "WhatsApp Access"],
        "all1000": ["MSISDN", "Recurring Charge", "WhatsApp Access"],
        "HomeFiber100": ["Installation", "Recurring Charge", "Bandwidth", "Data Cap"],
    }
    
    # Return components for the reference entity, or default set if not found
    return entity_components.get(reference_entity, ["MSISDN", "Recurring Charge", "Validity Period"])


def format_entity_list(entities: List[str]) -> str:
    """
    Formats a list of entities for display.
    """
    if not entities:
        return "No components available for modification."
        
    return "\n".join(f"• {entity}" for entity in entities)

def process_clone_modify_offer(user_input: str, session_context: dict) -> Tuple[str, dict]:
    """
    Enhanced process for cloning and modifying an existing offer.
    Now with better state management and error recovery.
    """
    # Get existing workflow data or create new
    workflow_data = session_context.get('workflow_data', {})
    workflow_state = session_context.get('workflow_state', 'COLLECT_EXISTING_OFFER')
    
    # Check if we're in modification mode for a completed workflow
    if session_context.get('workflow_completed', False) and ('CONFIRMED' in workflow_data or workflow_data.get('NEW_OFFER_NAME') != "NONE"):
        # Handle modifications to a completed workflow
        return handle_completed_workflow_modification("1", user_input, workflow_data)
    
    # If this is an empty dict, we're starting a new workflow
    if not workflow_data:
        # New workflow - extract initial information with LLM
        prompt = clone_modify_prompt.format(user_input=user_input)
        raw_response = invoke_bedrock_model(prompt=prompt, max_tokens=500, temperature=0.2)
        
        try:
            workflow_data = json.loads(raw_response)
            
            # Ensure all required fields exist
            for field in WORKFLOW_SCHEMAS["1"]["required_fields"]:
                if field not in workflow_data:
                    workflow_data[field] = "NONE"
            
            # Set conversation field if missing
            if "CONVERSATION" not in workflow_data:
                workflow_data["CONVERSATION"] = None
            
            # Special handling for the first interaction
            if workflow_data["EXISTING_OFFER_NAME"] != "NONE":
                existing_offer = workflow_data["EXISTING_OFFER_NAME"]
                
                # If we've identified an offer but don't have all details, start gathering them
                if workflow_data["NEW_OFFER_NAME"] == "NONE":
                    workflow_data["CONVERSATION"] = f"What would you like to name the new offer based on {existing_offer}?"
                    session_context['workflow_state'] = 'COLLECT_NEW_NAME'
                elif workflow_data["CATEGORY"] == "NONE":
                    workflow_data["CONVERSATION"] = f"What category should the new {workflow_data['NEW_OFFER_NAME']} offer be placed in?"
                    session_context['workflow_state'] = 'COLLECT_CATEGORY'
                elif workflow_data["ENTITIES_TO_MODIFY"] == "NONE":
                    # Show available entities for the reference offer
                    available_entities = get_available_entities(existing_offer)
                    workflow_data["AVAILABLE_ENTITIES"] = available_entities
                    workflow_data["CONVERSATION"] = (
                        f"Which components of the {existing_offer} offer would you like to modify? Available components:\n\n"
                        f"{format_entity_list(available_entities)}"
                    )
                    session_context['workflow_state'] = 'COLLECT_ENTITIES'
                elif workflow_data["CHANGES"] == "NONE":
                    workflow_data["CONVERSATION"] = (
                        f"What specific changes would you like to make to the {workflow_data['ENTITIES_TO_MODIFY']} in this new offer?"
                    )
                    session_context['workflow_state'] = 'COLLECT_CHANGES'
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from LLM response: {raw_response}")
            workflow_data = {
                "EXISTING_OFFER_NAME": "NONE",
                "NEW_OFFER_NAME": "NONE",
                "CATEGORY": "NONE",
                "ENTITIES_TO_MODIFY": "NONE",
                "CHANGES": "NONE",
                "CONVERSATION": "I couldn't understand your request. Could you tell me which existing offer you'd like to base your new offer on?"
            }
            session_context['workflow_state'] = 'COLLECT_EXISTING_OFFER'
    else:
        # We're continuing a conversation - process based on current state
        if workflow_state == 'COLLECT_EXISTING_OFFER':
            # Validate offer name
            if user_input.strip() in ["", "NONE", "none"]:
                workflow_data["CONVERSATION"] = "I need a valid offer name to continue. Please provide the name of an existing offer."
            else:
                workflow_data["EXISTING_OFFER_NAME"] = user_input
                workflow_data["CONVERSATION"] = f"What would you like to name the new offer based on {user_input}?"
                session_context['workflow_state'] = 'COLLECT_NEW_NAME'
                
        elif workflow_state == 'COLLECT_NEW_NAME':
            if user_input.strip() in ["", "NONE", "none"]:
                workflow_data["CONVERSATION"] = "Please provide a valid name for your new offer."
            else:
                workflow_data["NEW_OFFER_NAME"] = user_input
                workflow_data["CONVERSATION"] = f"What category should the new {user_input} offer be placed in?"
                session_context['workflow_state'] = 'COLLECT_CATEGORY'
                
        elif workflow_state == 'COLLECT_CATEGORY':
            if user_input.strip() in ["", "NONE", "none"]:
                workflow_data["CONVERSATION"] = "Please provide a valid category for your new offer."
            else:
                workflow_data["CATEGORY"] = user_input
                
                # Show available entities for modification
                existing_offer = workflow_data["EXISTING_OFFER_NAME"]
                available_entities = get_available_entities(existing_offer)
                workflow_data["AVAILABLE_ENTITIES"] = available_entities
                
                workflow_data["CONVERSATION"] = (
                    f"Which components of the {existing_offer} offer would you like to modify? Available components:\n\n"
                    f"{format_entity_list(available_entities)}"
                )
                session_context['workflow_state'] = 'COLLECT_ENTITIES'
                
        elif workflow_state == 'COLLECT_ENTITIES':
            # Check if user wants to see available entities
            if re.search(r"\b(?:show|list|what|available)\b.*\b(?:entities|components)\b", user_input, re.IGNORECASE):
                # Re-display the entities
                existing_offer = workflow_data["EXISTING_OFFER_NAME"]
                available_entities = get_available_entities(existing_offer)
                workflow_data["AVAILABLE_ENTITIES"] = available_entities
                
                workflow_data["CONVERSATION"] = (
                    f"Here are the components available in the {existing_offer} offer:\n\n"
                    f"{format_entity_list(available_entities)}\n\n"
                    f"Which ones would you like to modify?"
                )
            else:
                # User specified entities to modify
                workflow_data["ENTITIES_TO_MODIFY"] = user_input
                workflow_data["CONVERSATION"] = (
                    f"What specific changes would you like to make to the {user_input} in this new offer?"
                )
                session_context['workflow_state'] = 'COLLECT_CHANGES'
                
        elif workflow_state == 'COLLECT_CHANGES':
            workflow_data["CHANGES"] = user_input
            
            # After collecting all required information, analyze the changes
            analyze_changes_and_set_action(workflow_data)
            
            # Show confirmation of captured details
            workflow_data["CONVERSATION"] = generate_confirmation_message(workflow_data)
            session_context['workflow_state'] = 'CONFIRM_DETAILS'
            
        elif workflow_state == 'CONFIRM_DETAILS':
            # Check if user is confirming or correcting
            if re.search(r"\b(?:yes|correct|that's right|proceed|confirm)\b", user_input, re.IGNORECASE):
                # User confirms - mark as ready for final processing
                workflow_data["CONFIRMED"] = True
                session_context['workflow_state'] = 'COMPLETED'
            else:
                # Handle correction requests
                processed_correction = process_correction_request("1", user_input, workflow_data, session_context)
                if processed_correction:
                    workflow_data = processed_correction
    
    return json.dumps(workflow_data), workflow_data


def handle_completed_workflow_modification(workflow: str, user_input: str, workflow_data: dict) -> Tuple[str, dict]:
    """
    Handle modifications to an already completed workflow.
    """
    # First, check for mentions of new components/entities to add to the modification list
    new_entity = extract_entity_from_input(user_input)
                
    # If we found a new entity to add
    if new_entity:
        print(f"Detected new entity to modify: {new_entity}")
        
        # Add it to the existing entities
        current_entities = workflow_data.get("ENTITIES_TO_MODIFY", "")
        if current_entities and current_entities != "NONE":
            # If we already have entities, append the new one
            if new_entity.lower() not in current_entities.lower():  # Avoid duplicates
                workflow_data["ENTITIES_TO_MODIFY"] = f"{current_entities}, {new_entity}"
        else:
            # If no entities yet, set this as the first one
            workflow_data["ENTITIES_TO_MODIFY"] = new_entity
        
        # Ask for changes to this entity
        workflow_data["CONVERSATION"] = f"What changes would you like to make to the {new_entity}?"
        workflow_data.pop("CONFIRMED", None)  # Remove confirmation since we're making changes
        
        return json.dumps(workflow_data), workflow_data
        
    # Standard modification requests
    if workflow == "1":  # Clone-modify workflow
        if re.search(r"\b(name|rename|title|call it)\b", user_input, re.IGNORECASE):
            # User wants to change the name
            workflow_data['NEW_OFFER_NAME'] = "NONE"
            workflow_data['CONVERSATION'] = "What would you like to rename the offer to?"
            workflow_data.pop("CONFIRMED", None)
            
        elif re.search(r"\b(category|group|section)\b", user_input, re.IGNORECASE):
            # User wants to change the category
            workflow_data['CATEGORY'] = "NONE"
            workflow_data['CONVERSATION'] = "Which category would you like to place the offer in?"
            workflow_data.pop("CONFIRMED", None)
            
        elif re.search(r"\b(component|entity|modify|change)\b", user_input, re.IGNORECASE) and not new_entity:
            # User wants to change the components but didn't specify which one
            existing_offer = workflow_data.get('EXISTING_OFFER_NAME', '')
            available_entities = get_available_entities(existing_offer)
            
            workflow_data['CONVERSATION'] = (
                f"Which components would you like to modify? Available components in {existing_offer}:\n\n"
                f"{format_entity_list(available_entities)}\n\n"
                f"You can specify one or more components separated by commas."
            )
            workflow_data.pop("CONFIRMED", None)
            
        elif re.search(r"\b(change|adjustment|value|amount|limit)\b", user_input, re.IGNORECASE) and not new_entity:
            # User wants to change the specific changes but didn't specify which component
            entities = workflow_data.get('ENTITIES_TO_MODIFY', '')
            
            workflow_data['CONVERSATION'] = (
                f"For which component would you like to modify the changes? Current components being modified: {entities}\n\n"
                f"Please specify which component you want to change."
            )
            workflow_data.pop("CONFIRMED", None)
        
        # Check if this is a new change specification for existing entities
        elif "CHANGES" in workflow_data and workflow_data["CHANGES"] != "NONE":
            # If the input looks like a specification of changes
            if re.search(r"\b(increase|decrease|limit|set|change|add|remove|modify)\b", user_input, re.IGNORECASE):
                # Append as a new change
                workflow_data["CHANGES"] = f"{workflow_data['CHANGES']}; {user_input}"
                analyze_changes_and_set_action(workflow_data)
                
                # Ask for confirmation with the updated changes
                workflow_data["CONVERSATION"] = generate_confirmation_message(workflow_data)
        
        # If we couldn't figure out the user's intent - ask for clarification
        if "CONVERSATION" not in workflow_data or not workflow_data["CONVERSATION"]:
            workflow_data['CONVERSATION'] = (
                "I'm not sure what aspect of the offer you want to change. You can:\n\n"
                "• Change the offer name\n"
                "• Change the category\n"
                "• Add or modify components (like Data Allowance, Voice Minutes, etc.)\n"
                "• Specify changes to existing components\n\n"
                "Please tell me which of these you'd like to do."
            )
            workflow_data.pop("CONFIRMED", None)
    
    elif workflow == "2":  # Create entity workflow
        # Similar logic for entity workflow modifications
        # Check for name change request
        if re.search(r"\b(name|rename|title|call it)\b", user_input, re.IGNORECASE):
            workflow_data['NEW_ENTITY_NAME'] = "NONE"
            workflow_data['NEW_NAME_NEEDED'] = True
            workflow_data['CONVERSATION'] = f"What would you like to rename the {workflow_data.get('ENTITY_TYPE', 'entity')} to?"
            workflow_data.pop("CONFIRMED", None)
            
        # Check for category change request
        elif re.search(r"\b(category|group|section)\b", user_input, re.IGNORECASE):
            workflow_data['CATEGORY'] = "NONE"
            workflow_data['CONVERSATION'] = f"Which category would you like to place the {workflow_data.get('ENTITY_TYPE', 'entity')} in?"
            workflow_data.pop("CONFIRMED", None)
            
        # Check for subtype/template change request
        elif re.search(r"\b(subtype|template|type)\b", user_input, re.IGNORECASE):
            workflow_data['ENTITY_SUBTYPE'] = "NONE"
            workflow_data['BROWSING_STATE'] = "TEMPLATE_SELECTION"
            workflow_data['CURRENT_PAGE'] = 1
            workflow_data['CONVERSATION'] = get_entity_types_and_templates()
            workflow_data.pop("CONFIRMED", None)
            
        # Check for configuration change request
        elif re.search(r"\b(configuration|settings|parameters|details)\b", user_input, re.IGNORECASE):
            workflow_data['CONFIGURATION_CHANGES'] = "NONE"
            workflow_data['CONVERSATION'] = (
                f"What specific configurations would you like for this {workflow_data.get('ENTITY_TYPE', 'entity')}? "
                f"You can specify details like validity period, recurring charges, or any other parameters."
            )
            workflow_data.pop("CONFIRMED", None)
        
        # If we couldn't figure out the user's intent - ask for clarification
        if "CONVERSATION" not in workflow_data or not workflow_data["CONVERSATION"]:
            entity_type = workflow_data.get('ENTITY_TYPE', 'entity')
            workflow_data['CONVERSATION'] = (
                f"I'm not sure what aspect of the {entity_type} you want to change. You can:\n\n"
                f"• Change the {entity_type} name\n"
                f"• Change the category\n"
                f"• Change the template or subtype\n"
                f"• Modify the configuration details\n\n"
                f"Please tell me which of these you'd like to do."
            )
            workflow_data.pop("CONFIRMED", None)
    
    return json.dumps(workflow_data), workflow_data


def process_correction_request(workflow: str, user_input: str, workflow_data: dict, session_context: dict) -> Optional[dict]:
    """
    Process a request to correct a specific aspect of the workflow data.
    Returns updated workflow_data or None if not a correction request.
    """
    # Check for specific correction requests
    if workflow == "1":  # Clone-modify workflow
        if re.search(r"\b(?:change|update|modify|correct)\b.*\b(?:name|offer name)\b", user_input, re.IGNORECASE):
            workflow_data["NEW_OFFER_NAME"] = "NONE"
            workflow_data["CONVERSATION"] = "What would you like to rename the new offer to?"
            session_context['workflow_state'] = 'COLLECT_NEW_NAME'
            return workflow_data
        
        elif re.search(r"\b(?:change|update|modify|correct)\b.*\b(?:category)\b", user_input, re.IGNORECASE):
            workflow_data["CATEGORY"] = "NONE"
            workflow_data["CONVERSATION"] = "What category should the offer be placed in instead?"
            session_context['workflow_state'] = 'COLLECT_CATEGORY'
            return workflow_data
        
        elif re.search(r"\b(?:change|update|modify|correct)\b.*\b(?:components|entities)\b", user_input, re.IGNORECASE):
            workflow_data["ENTITIES_TO_MODIFY"] = "NONE"
            
            # Show available entities again
            existing_offer = workflow_data["EXISTING_OFFER_NAME"]
            available_entities = get_available_entities(existing_offer)
            
            workflow_data["CONVERSATION"] = (
                f"Which components would you like to modify instead? Available components:\n\n"
                f"{format_entity_list(available_entities)}"
            )
            session_context['workflow_state'] = 'COLLECT_ENTITIES'
            return workflow_data
        
        elif re.search(r"\b(?:change|update|modify|correct)\b.*\b(?:changes|modifications)\b", user_input, re.IGNORECASE):
            workflow_data["CHANGES"] = "NONE"
            workflow_data["CONVERSATION"] = (
                f"What changes would you like to make to the {workflow_data['ENTITIES_TO_MODIFY']} instead?"
            )
            session_context['workflow_state'] = 'COLLECT_CHANGES'
            return workflow_data
    
    elif workflow == "2":  # Create entity workflow
        # Similar logic for entity workflow corrections
        pass
    
    # Not a correction request
    return None

def process_create_entity(user_input: str, session_context: dict) -> Tuple[str, dict]:
    """
    Enhanced process for creating a new entity with better flow management.
    """
    # Get existing workflow data or create new
    workflow_data = session_context.get('workflow_data', {})
    workflow_state = session_context.get('workflow_state', 'COLLECT_ENTITY_TYPE')
    
    # Handle special workflow states with transitions
    if "BROWSING_STATE" in workflow_data:
        # User is in entity browsing mode
        result = handle_browsing_state(user_input, workflow_data, session_context)
        if result:
            return json.dumps(result), result
    
    # Regular workflow processing (not in browsing mode)
    if not workflow_data:
        # New workflow - use LLM to extract initial information
        prompt = create_entity_prompt.format(user_input=user_input)
        raw_response = invoke_bedrock_model(prompt=prompt, max_tokens=500, temperature=0.2)
        
        try:
            workflow_data = json.loads(raw_response)
            
            # Ensure we have all required fields with default values
            for field in WORKFLOW_SCHEMAS["2"]["required_fields"]:
                if field not in workflow_data:
                    workflow_data[field] = "NONE"
            
            # Set conversation field if missing
            if "CONVERSATION" not in workflow_data:
                workflow_data["CONVERSATION"] = None
            
            # If we have an entity type but no subtype/templates, enter browsing mode
            if workflow_data["ENTITY_TYPE"] != "NONE" and workflow_data["ENTITY_SUBTYPE"] == "NONE":
                workflow_data["BROWSING_STATE"] = "TEMPLATE_SELECTION"
                workflow_data["CURRENT_PAGE"] = 1
                workflow_data["CONVERSATION"] = get_entity_types_and_templates()
                session_context['workflow_state'] = 'BROWSE_TEMPLATES'
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from LLM response: {raw_response}")
            workflow_data = {
                "ENTITY_TYPE": "NONE",
                "ENTITY_SUBTYPE": "NONE",
                "CATEGORY": "NONE", 
                "BASE_ON_EXISTING": "NONE",
                "CONFIGURATION_CHANGES": "NONE",
                "CONVERSATION": "I couldn't understand your request. Could you please tell me what type of entity you'd like to create? (Package, Bundle, Promotion, Component, or Component Group)"
            }
            session_context['workflow_state'] = 'COLLECT_ENTITY_TYPE'
    else:
        # We're continuing an existing workflow - update based on current state
        if workflow_state == 'COLLECT_ENTITY_TYPE':
            # Check if input contains a valid entity type
            entity_types = ["package", "bundle", "promotion", "component", "component group"]
            entity_type_found = False
            
            for entity_type in entity_types:
                if re.search(fr"\b{entity_type}\b", user_input.lower()):
                    workflow_data["ENTITY_TYPE"] = entity_type.title()
                    entity_type_found = True
                    
                    # After getting entity type, prompt for templates/subtypes via browse
                    workflow_data["BROWSING_STATE"] = "TEMPLATE_SELECTION"
                    workflow_data["CURRENT_PAGE"] = 1
                    workflow_data["CONVERSATION"] = get_entity_types_and_templates()
                    session_context['workflow_state'] = 'BROWSE_TEMPLATES'
                    break
            
            # If no valid entity type found
            if not entity_type_found:
                workflow_data["CONVERSATION"] = (
                    "I'm sorry, I couldn't identify a valid entity type. Please select one of the following: "
                    "Package, Bundle, Promotion, Component, or Component Group."
                )
        
        elif workflow_state == 'COLLECT_ENTITY_SUBTYPE':
            # Check if user wants to browse templates
            if re.search(r"\b(?:show|list|browse|see|view|what|options)\b", user_input.lower()):
                workflow_data["BROWSING_STATE"] = "TEMPLATE_SELECTION" 
                workflow_data["CURRENT_PAGE"] = 1
                workflow_data["CONVERSATION"] = get_entity_types_and_templates()
                session_context['workflow_state'] = 'BROWSE_TEMPLATES'
            else:
                # User provided a specific template
                workflow_data["ENTITY_SUBTYPE"] = user_input
                
                # Move to category selection
                workflow_data["BROWSING_STATE"] = "CATEGORY_SELECTION"
                workflow_data["CURRENT_PAGE"] = 1
                workflow_data["CONVERSATION"] = get_category_list(workflow_data["ENTITY_TYPE"], workflow_data["ENTITY_SUBTYPE"])
                session_context['workflow_state'] = 'BROWSE_CATEGORIES'
        
        elif workflow_state == 'COLLECT_CATEGORY':
            # Check if user wants to browse categories
            if re.search(r"\b(?:show|list|browse|see|view|what|options)\b", user_input.lower()):
                workflow_data["BROWSING_STATE"] = "CATEGORY_SELECTION"
                workflow_data["CURRENT_PAGE"] = 1
                workflow_data["CONVERSATION"] = get_category_list(workflow_data["ENTITY_TYPE"], workflow_data["ENTITY_SUBTYPE"])
                session_context['workflow_state'] = 'BROWSE_CATEGORIES'
            else:
                # User provided a specific category
                workflow_data["CATEGORY"] = user_input
                
                # Ask about basing on existing
                if "BASE_ON_EXISTING" not in workflow_data or workflow_data["BASE_ON_EXISTING"] == "NONE":
                    workflow_data["CONVERSATION"] = (
                        f"Would you like to base this new {workflow_data['ENTITY_TYPE'].lower()} on an existing one, "
                        f"or create it from scratch? You can also say 'browse' to see available options."
                    )
                    session_context['workflow_state'] = 'COLLECT_BASE_PREFERENCE'
        
        elif workflow_state == 'COLLECT_BASE_PREFERENCE':
            # Process response about basing on existing
            if re.search(r"\b(?:yes|yeah|yep|sure|existing|base|reference)\b", user_input.lower()):
                workflow_data["BASE_ON_EXISTING"] = "YES"
                # Enter browsing mode for base selection
                workflow_data["BROWSING_STATE"] = "BASE_SELECTION"
                workflow_data["CURRENT_PAGE"] = 1
                workflow_data["CONVERSATION"] = get_entity_list_by_template_category(
                    workflow_data["ENTITY_TYPE"], 
                    workflow_data["ENTITY_SUBTYPE"], 
                    workflow_data["CATEGORY"]
                )
                session_context['workflow_state'] = 'BROWSE_ENTITIES'
            elif re.search(r"\b(?:no|nope|scratch|new|fresh)\b", user_input.lower()):
                workflow_data["BASE_ON_EXISTING"] = "NO"
                # For scratch creation, prompt for name
                workflow_data["CONVERSATION"] = f"What would you like to name this new {workflow_data['ENTITY_TYPE'].lower()}?"
                workflow_data["NEW_NAME_NEEDED"] = True
                session_context['workflow_state'] = 'COLLECT_NAME'
            elif re.search(r"\b(?:browse|show|list|see|available|existing)\b", user_input.lower()):
                workflow_data["BASE_ON_EXISTING"] = "BROWSE"
                # Enter browsing mode for base selection
                workflow_data["BROWSING_STATE"] = "BASE_SELECTION"
                workflow_data["CURRENT_PAGE"] = 1
                workflow_data["CONVERSATION"] = get_entity_list_by_template_category(
                    workflow_data["ENTITY_TYPE"], 
                    workflow_data["ENTITY_SUBTYPE"], 
                    workflow_data["CATEGORY"]
                )
                session_context['workflow_state'] = 'BROWSE_ENTITIES'
        
        elif workflow_state == 'COLLECT_NAME':
            # User is providing a name for the new entity
            if user_input.strip() in ["", "NONE", "none"]:
                workflow_data["CONVERSATION"] = f"Please provide a valid name for your new {workflow_data['ENTITY_TYPE'].lower()}."
            else:
                workflow_data["NEW_ENTITY_NAME"] = user_input
                workflow_data.pop("NEW_NAME_NEEDED", None)
                
                # Ask for configuration details if needed
                if "CONFIGURATION_CHANGES" not in workflow_data or workflow_data["CONFIGURATION_CHANGES"] == "NONE":
                    workflow_data["CONVERSATION"] = (
                        f"What specific configurations would you like for this new {workflow_data['ENTITY_TYPE'].lower()}? "
                        f"You can specify details like validity period, recurring charges, or any other parameters."
                    )
                    session_context['workflow_state'] = 'COLLECT_CONFIG'
        
        elif workflow_state == 'COLLECT_CONFIG':
            # User is providing configuration details
            workflow_data["CONFIGURATION_CHANGES"] = user_input
            
            # Show confirmation of captured details
            workflow_data["CONVERSATION"] = generate_entity_confirmation_message(workflow_data)
            session_context['workflow_state'] = 'CONFIRM_DETAILS'
        
        elif workflow_state == 'COLLECT_ENTITIES_TO_MODIFY':
            # User is specifying which entities to modify after selecting a base
            workflow_data["ENTITIES_TO_MODIFY"] = user_input
            
            # Ask for specific changes
            workflow_data["CONVERSATION"] = (
                f"What changes would you like to make to these components? Please specify details like "
                f"cardinality changes (e.g., 'limit to 2 MSISDNs'), value adjustments (e.g., 'increase recurring charge by 10%'), "
                f"or configuration changes (e.g., 'add 10GB data allowance')."
            )
            session_context['workflow_state'] = 'COLLECT_CHANGES'
        
        elif workflow_state == 'COLLECT_CHANGES':
            # User is specifying changes to the selected entities
            workflow_data["CHANGES"] = user_input
            
            # Ask for new entity name if not already provided
            if "NEW_ENTITY_NAME" not in workflow_data:
                workflow_data["CONVERSATION"] = f"What would you like to name this new {workflow_data['ENTITY_TYPE'].lower()}?"
                workflow_data["NEW_NAME_NEEDED"] = True
                session_context['workflow_state'] = 'COLLECT_NAME'
            else:
                # All required info collected - workflow complete
                analyze_changes_and_set_action(workflow_data)
                
                # Show confirmation of captured details
                workflow_data["CONVERSATION"] = generate_entity_confirmation_message(workflow_data)
                session_context['workflow_state'] = 'CONFIRM_DETAILS'
        
        elif workflow_state == 'CONFIRM_DETAILS':
            # Check if user is confirming or rejecting
            if re.search(r"\b(?:yes|correct|that's right|proceed|confirm)\b", user_input, re.IGNORECASE):
                # User confirms - mark as ready for final processing
                workflow_data["CONFIRMED"] = True
                session_context['workflow_state'] = 'COMPLETED'
            else:
                # Handle correction requests
                processed_correction = process_correction_request("2", user_input, workflow_data, session_context)
                if processed_correction:
                    workflow_data = processed_correction
    
    return json.dumps(workflow_data), workflow_data


def handle_browsing_state(user_input: str, workflow_data: dict, session_context: dict) -> Optional[dict]:
    """
    Handle user input when in a browsing state.
    Returns updated workflow_data if handled, None otherwise.
    """
    browsing_state = workflow_data["BROWSING_STATE"]
    
    # Check if user wants to exit browsing
    exit_patterns = [
        r"\b(?:cancel|exit|quit|stop)\b.*\b(?:browsing|looking|searching)\b",
        r"\b(?:go|start)\b.*\b(?:over|again|back)\b",
        r"\b(?:different|another)\b.*\b(?:approach|way|method)\b",
        r"^exit browsing$",
        r"^exit$",
        r"^cancel$"
    ]
    
    if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in exit_patterns):
        # Exit browsing and reset to appropriate state
        workflow_data.pop("BROWSING_STATE", None)
        workflow_data.pop("CURRENT_PAGE", None)
        
        # Reset to appropriate state based on what we were browsing
        if browsing_state == "TEMPLATE_SELECTION":
            workflow_data["CONVERSATION"] = (
                f"Let's try a different approach. What type of template would you like to use for your "
                f"{workflow_data.get('ENTITY_TYPE', 'entity')}? You can specify directly."
            )
            session_context['workflow_state'] = 'COLLECT_ENTITY_SUBTYPE'
        elif browsing_state == "CATEGORY_SELECTION":
            workflow_data["CONVERSATION"] = (
                f"Let's try a different approach. What category would you like to place your "
                f"{workflow_data.get('ENTITY_TYPE', 'entity')} in? You can specify directly."
            )
            session_context['workflow_state'] = 'COLLECT_CATEGORY'
        elif browsing_state == "BASE_SELECTION":
            workflow_data["CONVERSATION"] = (
                f"Would you like to base this on an existing {workflow_data.get('ENTITY_TYPE', 'entity')} "
                f"or create it from scratch?"
            )
            session_context['workflow_state'] = 'COLLECT_BASE_PREFERENCE'
        
        return workflow_data
    
    # Check if user wants to see more items
    if re.search(r"\b(?:more|next|continue|show more)\b", user_input, re.IGNORECASE):
        # Show the next page of results
        page = workflow_data.get("CURRENT_PAGE", 1) + 1
        workflow_data["CURRENT_PAGE"] = page
        
        if browsing_state == "TEMPLATE_SELECTION":
            workflow_data["CONVERSATION"] = get_entity_types_and_templates(page=page)
        elif browsing_state == "CATEGORY_SELECTION":
            workflow_data["CONVERSATION"] = get_category_list(
                workflow_data["ENTITY_TYPE"], 
                workflow_data["ENTITY_SUBTYPE"], 
                page=page
            )
        elif browsing_state == "BASE_SELECTION":
            workflow_data["CONVERSATION"] = get_entity_list_by_template_category(
                workflow_data["ENTITY_TYPE"], 
                workflow_data["ENTITY_SUBTYPE"], 
                workflow_data["CATEGORY"], 
                page=page
            )
        
        return workflow_data
    
    # Check if user is making a selection from the browse list
    selection_match = re.search(r"\b(?:select|choose|use|pick)\b.*?(\d+|[a-zA-Z0-9]+)", user_input, re.IGNORECASE)
    if selection_match or re.match(r"^\d+$", user_input.strip()):
        # Extract the selection
        selected_item = selection_match.group(1) if selection_match else user_input.strip()
        
        # Look up the actual entity name from available options
        if browsing_state == "TEMPLATE_SELECTION":
            selected_entity = get_entity_by_selection(selected_item, "Template")
            workflow_data["ENTITY_SUBTYPE"] = selected_entity
            workflow_data["BROWSING_STATE"] = "CATEGORY_SELECTION"
            workflow_data["CURRENT_PAGE"] = 1
            workflow_data["CONVERSATION"] = get_category_list(workflow_data["ENTITY_TYPE"], selected_entity)
            session_context['workflow_state'] = 'BROWSE_CATEGORIES'
            return workflow_data
        
        elif browsing_state == "CATEGORY_SELECTION":
            selected_entity = get_entity_by_selection(selected_item, "Category")
            workflow_data["CATEGORY"] = selected_entity
            workflow_data["BROWSING_STATE"] = "BASE_SELECTION"
            workflow_data["CURRENT_PAGE"] = 1
            workflow_data["CONVERSATION"] = get_entity_list_by_template_category(
                workflow_data["ENTITY_TYPE"], 
                workflow_data["ENTITY_SUBTYPE"], 
                selected_entity
            )
            session_context['workflow_state'] = 'BROWSE_ENTITIES'
            return workflow_data
        
        elif browsing_state == "BASE_SELECTION":
            # Check if user wants to create new instead of selecting
            if re.search(r"\b(?:create|make|new)\b.*\b(?:new|fresh|scratch)\b", user_input, re.IGNORECASE):
                workflow_data["BASE_ON_EXISTING"] = "NO"
                workflow_data.pop("BROWSING_STATE", None)
                workflow_data.pop("CURRENT_PAGE", None)
                
                # Prompt for name
                workflow_data["CONVERSATION"] = f"What would you like to name this new {workflow_data['ENTITY_TYPE'].lower()}?"
                workflow_data["NEW_NAME_NEEDED"] = True
                session_context['workflow_state'] = 'COLLECT_NAME'
                return workflow_data
            
            # User has selected a base entity
            selected_entity = get_entity_by_selection(selected_item, workflow_data["ENTITY_TYPE"])
            workflow_data["BASE_ON_EXISTING"] = "YES"
            workflow_data["REFERENCE_ENTITY"] = selected_entity
            workflow_data.pop("BROWSING_STATE", None)  # Exit browsing mode
            workflow_data.pop("CURRENT_PAGE", None)
            
            # Get all modifiable entities from the reference package
            workflow_data["AVAILABLE_ENTITIES"] = get_available_entities(selected_entity)
            workflow_data["CONVERSATION"] = (
                f"Great! You've selected **{selected_entity}** as your base {workflow_data['ENTITY_TYPE'].lower()}. "
                f"Which components would you like to modify? Available components:\n\n"
                f"{format_entity_list(workflow_data['AVAILABLE_ENTITIES'])}"
            )
            session_context['workflow_state'] = 'COLLECT_ENTITIES_TO_MODIFY'
            return workflow_data
    
    # Not handled
    return None


def generate_entity_confirmation_message(workflow_data: dict) -> str:
    """
    Generate a confirmation message for entity creation.
    """
    entity_type = workflow_data.get('ENTITY_TYPE', 'Entity')
    entity_subtype = workflow_data.get('ENTITY_SUBTYPE', 'Default Type')
    category = workflow_data.get('CATEGORY', 'Default Category')
    base_on = workflow_data.get('BASE_ON_EXISTING', 'NO')
    
    # Different format based on creation mode
    if base_on in ["YES", "BROWSE"] and "REFERENCE_ENTITY" in workflow_data:
        # Using an existing entity as base
        reference = workflow_data.get("REFERENCE_ENTITY", "an existing template")
        entities = workflow_data.get("ENTITIES_TO_MODIFY", "standard components")
        changes = workflow_data.get("CHANGES", "default settings")
        entity_name = workflow_data.get('NEW_ENTITY_NAME', f'New {entity_type}')
        
        # Format changes as bullet points if available
        formatted_changes = ""
        if changes and changes != "default settings":
            for change in changes.split(';'):
                change = change.strip()
                if change:
                    formatted_changes += f"  • {change}\n"
        
        # Safe alternative to avoid f-string backslash issue
        default_changes = '  • Using default settings from the template\n'
        changes_to_display = formatted_changes if formatted_changes else default_changes
        
        return (
            f"**{entity_type} Creation Summary**\n\n"
            f"I'll create a new {entity_type.lower()} with these details:\n\n"
            f"• Base {entity_type.lower()}: **{reference}**\n"
            f"• New name: **{entity_name}**\n"
            f"• Template: **{entity_subtype}**\n"
            f"• Category: **{category}**\n"
            f"• Components to modify: **{entities}**\n"
            f"• Changes to apply:\n{changes_to_display}\n"
            f"Is this correct? Or would you like to make any changes?"
        )
    else:
        # Creating from scratch
        entity_name = workflow_data.get('NEW_ENTITY_NAME', f'New {entity_type}')
        config_changes = workflow_data.get("CONFIGURATION_CHANGES", "default configuration")
        
        return (
            f"**{entity_type} Creation Summary**\n\n"
            f"I'll create a new {entity_type.lower()} with these details:\n\n"
            f"• Name: **{entity_name}**\n"
            f"• Template: **{entity_subtype}**\n"
            f"• Category: **{category}**\n"
            f"• Configuration: **{config_changes}**\n\n"
            f"Is this correct? Or would you like to make any changes?"
        )

def get_bedrock_client():
    """Lazily initialize Bedrock client to reduce cold start time."""
    global bedrock_client
    if bedrock_client is None:
        bedrock_client = boto3.client('bedrock-runtime')
    return bedrock_client


def get_dynamodb_client():
    """Lazily initialize DynamoDB client to reduce cold start time."""
    global dynamodb_client
    if dynamodb_client is None:
        dynamodb_client = boto3.client('dynamodb')
    return dynamodb_client


def get_api_gateway_management_client(domain_name, stage):
    """
    Get or create API Gateway Management client.
    Uses caching to avoid creating new clients for the same endpoint.
    """
    global apigw_management_client
    endpoint_url = f'https://{domain_name}/{stage}'
    
    # Always create a new client with the correct endpoint
    # This ensures we're using the right connection details
    apigw_management_client = boto3.client(
        'apigatewaymanagementapi', 
        endpoint_url=endpoint_url
    )
    return apigw_management_client


def get_session_context(session_id: str) -> Dict:
    """
    Get session context from cache or DynamoDB with improved error handling.
    """
    # Check cache first
    if session_id in _session_cache:
        cached_data = _session_cache[session_id]
        cache_age = (datetime.now() - datetime.fromisoformat(cached_data['timestamp'])).total_seconds()
        if cache_age < CACHE_TTL:
            return cached_data['data']
    
    try:
        # Retrieve from DynamoDB
        response = get_dynamodb_client().get_item(
            TableName=SESSION_HISTORY_TABLE,
            Key={'sessionId': {'S': session_id}}
        )
        
        # Process response
        if 'Item' in response:
            item = response['Item']
            
            # Parse JSON fields safely
            try:
                conversation_history = json.loads(item.get('conversationHistory', {}).get('S', '[]'))
            except json.JSONDecodeError:
                print(f"Failed to parse conversation history for session {session_id}")
                conversation_history = []
            
            try:
                workflow_data = json.loads(item.get('workflowData', {}).get('S', '{}'))
            except json.JSONDecodeError:
                print(f"Failed to parse workflow data for session {session_id}")
                workflow_data = {}
            
            context = {
                'sessionId': item['sessionId']['S'],
                'conversation_history': conversation_history,
                'current_workflow': item.get('currentWorkflow', {}).get('S', '0'),
                'workflow_data': workflow_data,
                'customer_id': item.get('customerId', {}).get('S', 'unknown'),
                'last_updated': item.get('lastUpdated', {}).get('S', datetime.now().isoformat())
            }
            
            # Add workflow_completed flag if it exists
            if 'workflowCompleted' in item:
                context['workflow_completed'] = item['workflowCompleted']['BOOL']
            
            # Add workflow_state if it exists
            if 'workflowState' in item:
                context['workflow_state'] = item['workflowState']['S']
            
            # Update cache
            _session_cache[session_id] = {
                'timestamp': datetime.now().isoformat(),
                'data': context
            }
            
            return context
        else:
            # New session
            new_context = {
                'sessionId': session_id,
                'conversation_history': [],
                'current_workflow': '0',
                'workflow_data': {},
                'customer_id': 'unknown',
                'last_updated': datetime.now().isoformat()
            }
            
            # Update cache
            _session_cache[session_id] = {
                'timestamp': datetime.now().isoformat(),
                'data': new_context
            }
            
            return new_context
            
    except Exception as e:
        print(f"Error retrieving session context: {str(e)}")
        traceback.print_exc()
        
        # Return a fresh context as fallback
        return {
            'sessionId': session_id,
            'conversation_history': [],
            'current_workflow': '0',
            'workflow_data': {},
            'customer_id': 'unknown',
            'last_updated': datetime.now().isoformat()
        }


def update_session_history(
    session_id: str, 
    customer_id: str, 
    query: str, 
    workflow: str, 
    response: str, 
    workflow_data: Optional[Dict] = None
) -> None:
    """
    Enhanced session history update with better error handling and context management.
    """
    try:
        # Get current context
        context = get_session_context(session_id)
        
        # Handle conversation reset for general queries
        if is_conversational_input(query) and workflow == "0":
            # Keep limited history for context but reset workflow data
            context['conversation_history'] = context.get('conversation_history', [])[-5:]
            context['workflow_data'] = {}
            context['current_workflow'] = "0"
            if 'workflow_completed' in context:
                context.pop('workflow_completed')
            if 'workflow_state' in context:
                context.pop('workflow_state')
        
        # Update conversation history
        conversation_history = context.get('conversation_history', [])
        
        # Add the new exchange
        conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'workflow': workflow
        })
        
        # Limit history size but keep enough for context
        if len(conversation_history) > 30:
            conversation_history = conversation_history[-30:]
        
        # Update workflow data if provided
        if workflow_data:
            # Special handling for workflow transitions
            if workflow == '0' and context.get('current_workflow') != '0':
                # We've exited a workflow - clear the workflow data
                context['workflow_data'] = {}
                context.pop('workflow_completed', None)
                context.pop('workflow_state', None)
            else:
                context['workflow_data'] = workflow_data
        
        # Special handling for workflow completion flag
        if workflow != '0' and context.get('current_workflow') == workflow:
            # Preserve the workflow_completed flag
            workflow_completed = context.get('workflow_completed', False)
        else:
            workflow_completed = False
            
        # Update current workflow
        context['current_workflow'] = workflow
        context['customer_id'] = customer_id
        
        # Keep workflow_completed flag if present
        if workflow_completed:
            context['workflow_completed'] = workflow_completed
        
        # Update session timestamp
        context['last_updated'] = datetime.now().isoformat()
        
        # Prepare DynamoDB Item
        item = {
            'sessionId': {'S': session_id},
            'customerId': {'S': customer_id},
            'currentWorkflow': {'S': workflow},
            'lastUpdated': {'S': datetime.now().isoformat()},
            'conversationHistory': {'S': json.dumps(conversation_history)},
            'workflowData': {'S': json.dumps(context.get('workflow_data', {}))}
        }
        
        # Add workflow_completed if it exists
        if 'workflow_completed' in context:
            item['workflowCompleted'] = {'BOOL': context['workflow_completed']}
        
        # Add workflow_state if it exists
        if 'workflow_state' in context:
            item['workflowState'] = {'S': context['workflow_state']}
        
        # Update DynamoDB
        get_dynamodb_client().put_item(
            TableName=SESSION_HISTORY_TABLE,
            Item=item
        )
        
        # Update cache
        _session_cache[session_id] = {
            'timestamp': datetime.now().isoformat(),
            'data': context
        }
        
    except Exception as e:
        print(f"Error updating session history: {str(e)}")
        traceback.print_exc()


def send_response(connection_id: str, domain_name: str, stage: str, payload: Dict) -> None:
    """
    Send response back to client over WebSocket connection with enhanced error handling.
    """
    try:
        print(f"Sending response to connection: {connection_id}")
        
        # Ensure the payload is serializable
        try:
            serialized_payload = json.dumps(payload)
        except TypeError as e:
            print(f"Error serializing payload: {str(e)}")
            # Replace non-serializable values with strings
            sanitized_payload = sanitize_json(payload)
            serialized_payload = json.dumps(sanitized_payload)
        
        # Get API Gateway client and send response
        api_client = get_api_gateway_management_client(domain_name, stage)
        api_client.post_to_connection(
            ConnectionId=connection_id,
            Data=serialized_payload.encode('utf-8')
        )
        
    except Exception as e:
        print(f"Error sending response to connection {connection_id}: {str(e)}")
        traceback.print_exc()
        
        # Handle GoneException (client disconnected)
        if hasattr(e, 'response') and e.response.get('Error', {}).get('Code') == 'GoneException':
            print(f"Connection {connection_id} is gone, removing from tracking")
            handle_disconnect(connection_id)


def sanitize_json(obj):
    """
    Recursively sanitize an object to ensure it can be JSON serialized.
    Converts non-serializable values to strings.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Convert anything else to string
        return str(obj)


def send_error(connection_id: str, domain_name: str, stage: str, error_message: str) -> None:
    """
    Send error message to client with improved formatting.
    """
    payload = {
        'type': 'error',
        'message': error_message,
        'timestamp': datetime.now().isoformat()
    }
    send_response(connection_id, domain_name, stage, payload)


def extract_entity_from_input(user_input: str) -> Optional[str]:
    """
    Extracts entity names from user input text with improved pattern matching.
    """
    # List of common telecom entity terms
    entity_terms = [
        "Data Allowance", "Voice Minutes", "SMS Count", "Validity Period", 
        "MSISDN", "Recurring Charge", "WhatsApp Access", "Data Pack", 
        "Roaming", "International Calling", "Premium Content",
        "Installation", "Bandwidth", "Data Cap", "Speed", "Contract Duration"
    ]
    
    # Check for exact matches first (with case-insensitive partial matching)
    for term in entity_terms:
        # Check both exact and possessive forms (e.g., "data allowance" and "data allowance's")
        if term.lower() in user_input.lower() or f"{term.lower()}'s" in user_input.lower():
            return term
    
    # Try to extract using patterns
    entity_patterns = [
        r"(?:add|include|change|modify).*(?:entity|component|item|field).*(?:called|named)\s+([a-zA-Z\s]+)",
        r"(?:also|add|include|change|modify)\s+(?:the\s+)?([a-zA-Z\s]+(?:allowance|charge|limit|usage|bundle|pack|duration|speed|cap|roaming))",
        r"(?:want to|like to|need to)\s+(?:change|modify|update)\s+([a-zA-Z\s]+(?:allowance|charge|limit|usage|bundle|pack|duration|speed|cap|roaming))",
        r"(?:the|with|for)\s+([a-zA-Z\s]+(?:allowance|charge|limit|usage|bundle|pack|duration|speed|cap|roaming))"
    ]
    
    for pattern in entity_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            entity = match.group(1).strip()
            # Check if extracted entity is too generic
            if len(entity) < 3 or entity.lower() in ['the', 'this', 'that', 'those', 'these', 'some', 'any']:
                continue
            return entity
    
    return None


def invoke_bedrock_model(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = 0.7) -> str:
    """
    Invoke Bedrock model with text prompt and improved error handling.
    """
    try:
        # Prepare request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": temperature
        }
        
        # Add timeout for API call
        response = get_bedrock_client().invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body['content'][0]['text']
        
    except Exception as e:
        print(f"Error invoking Bedrock model: {str(e)}")
        traceback.print_exc()
        return "I encountered an error processing your request. Please try again with simpler instructions."


def invoke_bedrock_model_with_messages(messages: List[Dict], max_tokens: int = MAX_TOKENS, 
                                       temperature: float = 0.7, system_prompt: str = None) -> str:
    """
    Invoke Bedrock model with structured messages and improved error handling.
    """
    try:
        # Prepare request body with structured messages
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": []
        }
        
        # Add system message if provided
        if system_prompt:
            request_body["system"] = system_prompt
            
        # Convert message format for Bedrock
        formatted_messages = []
        for msg in messages:
            # Ensure content is a string and not None or empty
            content = msg.get("content", "")
            if content is None or not isinstance(content, str):
                content = str(content) if content is not None else ""
                
            formatted_msg = {
                "role": msg["role"],
                "content": [{"type": "text", "text": content}]
            }
            formatted_messages.append(formatted_msg)
            
        request_body["messages"] = formatted_messages
        request_body["temperature"] = temperature
        
        # Print the request for debugging
        print(f"Bedrock request: {json.dumps(request_body)}")
        
        # Invoke model with timeout
        response = get_bedrock_client().invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body['content'][0]['text']
        
    except Exception as e:
        print(f"Error invoking Bedrock model with messages: {str(e)}")
        traceback.print_exc()
        
        # Provide a more helpful error message based on the type of exception
        if "RequestEntityTooLargeException" in str(e) or "RequestLimitExceeded" in str(e):
            return "I'm having trouble processing your request because it's too large. Could you try simplifying it or breaking it down into smaller parts?"
        elif "ServiceUnavailable" in str(e) or "InternalServerError" in str(e):
            return "I'm currently experiencing service issues. Please try again in a moment."
        elif "Throttling" in str(e) or "TooManyRequestsException" in str(e):
            return "I'm receiving too many requests right now. Please try again in a moment."
        elif "ValidationException" in str(e) and "messages" in str(e):
            return "I encountered an issue processing the conversation. Let's start fresh - what would you like help with?"
        else:
            return "I encountered an error processing your request. Please try again with simpler instructions."

def process_general_query(query: str, session_context: Dict) -> str:
    """
    Process general (non-catalog) queries using Bedrock Claude with improved context and error handling.
    """
    try:
        # System instructions for the model
        system_prompt = (
            "You are an AI Assistant for a telecom product manager. "
            "You help with catalog management for telecom products including offers, packages, bundles, and components. "
            "Keep responses concise, helpful, and professional. "
            "\n\n"
            "If asked about creating a new offer based on an existing one, help the user understand they can "
            "use expressions like 'Create an offer similar to X' or 'Clone X with these changes'."
            "\n\n"
            "If asked about creating a new entity from scratch, explain they can start with 'Create a new entity' "
            "and you'll guide them through the available types (Package, Bundle, Promotion, Component, Component Group)."
            "\n\n"
            "For catalog browsing, let users know they can ask to see available entities of different types."
            "\n\n"
            "Always maintain a conversational and friendly tone. If the user's intent isn't clear, "
            "politely ask clarifying questions."
        )

        # Build conversation history for context
        conversation_history = session_context.get('conversation_history', [])
        
        # Only use recent exchanges to stay within context limits
        recent_exchanges = conversation_history[-5:] if conversation_history else []

        messages = []
        for exchange in recent_exchanges:
            if 'query' in exchange and exchange['query']:
                user_message = exchange['query']
                if not isinstance(user_message, str):
                    user_message = str(user_message)
                messages.append({"role": "user", "content": user_message})
                
            if 'response' in exchange and exchange['response']:
                assistant_message = exchange['response']
                if not isinstance(assistant_message, str):
                    assistant_message = str(assistant_message)
                messages.append({"role": "assistant", "content": assistant_message})

        # Add the current query
        messages.append({"role": "user", "content": query})

        # If we have no messages, create a default introduction
        if len(messages) <= 1:
            return (
                "Hello! I'm here to help you manage your telecom product catalog. I can assist you with:\n\n"
                "- Creating or modifying offers, packages, bundles, and components\n"
                "- Browsing existing catalog items\n"
                "- Cloning and customizing existing products\n\n"
                "What would you like to do?"
            )

        # Invoke the model
        response = invoke_bedrock_model_with_messages(
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            system_prompt=system_prompt
        )
        return response

    except Exception as e:
        print(f"Error in process_general_query: {e}", traceback.format_exc())
        return (
            "Hello! I'm here to help you manage your telecom product catalog. I can assist you with:\n\n"
            "- Creating or modifying offers, packages, bundles, and components\n"
            "- Browsing existing catalog items\n"
            "- Cloning and customizing existing products\n\n"
            "What would you like to do?"
        )


def is_workflow_data_complete(workflow: str, data: dict) -> bool:
    """
    Enhanced check for workflow completion with improved validation.
    """
    if workflow not in WORKFLOW_SCHEMAS:
        return False
        
    schema = WORKFLOW_SCHEMAS[workflow]
    
    # Special case handling for different workflows
    if workflow == "1":  # Clone-modify workflow
        # Check for confirmation flag
        if "CONFIRMED" in data and data["CONFIRMED"]:
            return True
            
        # Check if conversation contains prompt for modification - not complete yet
        if "CONVERSATION" in data and data["CONVERSATION"]:
            conversation = data.get("CONVERSATION", "").lower()
            if any(phrase in conversation for phrase in [
                "what would you like to change",
                "what specific aspect",
                "which components would you like to modify",
                "what changes would you like to make",
                "what would you like to rename",
                "what category should"
            ]):
                return False
            
        # Check all required fields are filled
        for field in schema["required_fields"]:
            if field not in data or data[field] == "NONE" or not data[field]:
                return False
                
        # Check if in confirmation stage
        if "CONVERSATION" in data and data["CONVERSATION"] and (
            "correct" in data["CONVERSATION"].lower() or 
            "is this" in data["CONVERSATION"].lower() or
            "summary" in data["CONVERSATION"].lower()
        ):
            # We're waiting for confirmation, not complete yet
            return False
            
        return True
            
    elif workflow == "2":  # Create entity workflow
        # If we're in browsing state, we're not complete
        if "BROWSING_STATE" in data:
            return False
            
        # Check if user needs to provide a name
        if "NEW_NAME_NEEDED" in data and data["NEW_NAME_NEEDED"]:
            return False
            
        # Check if conversation contains prompt for modification - not complete yet
        if "CONVERSATION" in data and data["CONVERSATION"]:
            conversation = data.get("CONVERSATION", "").lower()
            if any(phrase in conversation for phrase in [
                "what would you like to change",
                "what specific aspect",
                "which components would you like to modify",
                "what changes would you like to make",
                "what would you like to rename",
                "what category should"
            ]):
                return False
            
        # For base-on-existing flow, check if changes have been specified
        if data.get("BASE_ON_EXISTING") in ["YES", "BROWSE"] and "REFERENCE_ENTITY" in data:
            if "ENTITIES_TO_MODIFY" not in data or data["ENTITIES_TO_MODIFY"] == "NONE":
                return False
            if "CHANGES" not in data or data["CHANGES"] == "NONE":
                return False
            if "NEW_ENTITY_NAME" not in data:
                return False
        
        # For from-scratch flow, check if name and config are specified
        if data.get("BASE_ON_EXISTING") == "NO":
            if "NEW_ENTITY_NAME" not in data:
                return False
            if "CONFIGURATION_CHANGES" not in data or data["CONFIGURATION_CHANGES"] == "NONE":
                return False
        
        # Check all required fields
        for field in schema["required_fields"]:
            if field not in data or data[field] == "NONE" or not data[field]:
                return False
        
        # Check for confirmation
        if "CONFIRMED" in data and data["CONFIRMED"]:
            return True
        
        # Check if in confirmation stage
        if "CONVERSATION" in data and data["CONVERSATION"] and (
            "correct" in data["CONVERSATION"].lower() or 
            "is this" in data["CONVERSATION"].lower() or
            "summary" in data["CONVERSATION"].lower()
        ):
            # We're waiting for confirmation, not complete yet
            return False
                
        return True
    
    # Generic case - just check all required fields
    for field in schema["required_fields"]:
        if field not in data or data[field] == "NONE" or not data[field]:
            return False
            
    return True


def generate_final_response(workflow: str, data: dict) -> str:
    """
    Enhanced final response generator with better formatting and clarity.
    """
    try:
        if workflow == "1":  # Clone-modify existing offer
            # Get nicely formatted details
            existing_offer = data.get('EXISTING_OFFER_NAME', 'Unknown Offer')
            new_offer = data.get('NEW_OFFER_NAME', 'New Offer')
            category = data.get('CATEGORY', 'Default Category')
            entities = data.get('ENTITIES_TO_MODIFY', 'No specified components')
            changes = data.get('CHANGES', 'No specified changes')
            
            # Convert semicolon-separated changes to bullet points
            formatted_changes = ""
            for change in changes.split(';'):
                change = change.strip()
                if change:
                    formatted_changes += f"  • {change}\n"
            
            if not formatted_changes:
                formatted_changes = "  • No specific changes\n"
            
            # Format response with more details and clear next steps
            return (
                f"## 📋 New Offer Created\n\n"
                f"I've created a new offer named **{new_offer}** based on **{existing_offer}** in the **{category}** category.\n\n"
                f"### Components Modified:\n"
                f"  • {entities}\n\n"
                f"### Changes Applied:\n"
                f"{formatted_changes}\n"
                f"### Next Steps\n"
                f"Your new offer has been created and is ready for review. Would you like to:\n\n"
                f"1. Make additional changes to this offer\n"
                f"2. Create another offer\n"
                f"3. Proceed to deployment\n\n"
                f"Please type the number of your choice (1, 2, or 3)."
            )
            
        elif workflow == "2":  # Create new entity
            # Get nicely formatted details
            entity_type = data.get('ENTITY_TYPE', 'Entity')
            entity_subtype = data.get('ENTITY_SUBTYPE', 'Default Type')
            category = data.get('CATEGORY', 'Default Category')
            base_on = data.get('BASE_ON_EXISTING', 'NO')
            entity_name = data.get('NEW_ENTITY_NAME', f'New {entity_type}')
            
            # Different responses based on creation mode
            if base_on in ["YES", "BROWSE"] and "REFERENCE_ENTITY" in data:
                # Using an existing entity as base
                reference = data.get("REFERENCE_ENTITY", "an existing template")
                entities = data.get("ENTITIES_TO_MODIFY", "standard components")
                changes = data.get("CHANGES", "default settings")
                
                # Format changes as bullet points if available
                formatted_changes = ""
                if changes and changes != "default settings":
                    for change in changes.split(';'):
                        change = change.strip()
                        if change:
                            formatted_changes += f"  • {change}\n"
                
                # Safe alternative to avoid f-string backslash issue
                default_changes = '  • Using default settings from the template\n'
                changes_to_display = formatted_changes if formatted_changes else default_changes
                
                return (
                    f"## 📋 New {entity_type} Created\n\n"
                    f"I've created a new **{entity_type}** named **{entity_name}** of type **{entity_subtype}** "
                    f"in the **{category}** category, based on **{reference}**.\n\n"
                    f"### Modified Components:\n"
                    f"  • {entities}\n\n"
                    f"### Changes Applied:\n"
                    f"{changes_to_display}"
                    f"### Next Steps\n"
                    f"Your new {entity_type.lower()} has been created and is ready for review. Would you like to:\n\n"
                    f"1. Make additional changes to this {entity_type.lower()}\n"
                    f"2. Create another entity\n"
                    f"3. Proceed to deployment\n\n"
                    f"Please type the number of your choice (1, 2, or 3)."
                )
            else:
                # Creating from scratch
                config_changes = data.get("CONFIGURATION_CHANGES", "default configuration")
                
                return (
                    f"## 📋 New {entity_type} Created\n\n"
                    f"I've created a new **{entity_type}** named **{entity_name}** of type **{entity_subtype}** "
                    f"in the **{category}** category from scratch.\n\n"
                    f"### Configuration Details:\n"
                    f"  • {config_changes}\n\n"
                    f"### Next Steps\n"
                    f"Your new {entity_type.lower()} has been created with default components and configuration. Would you like to:\n\n"
                    f"1. Customize the components of this {entity_type.lower()}\n"
                    f"2. Create another entity\n"
                    f"3. Proceed to deployment\n\n"
                    f"Please type the number of your choice (1, 2, or 3)."
                )
    
    except Exception as e:
        print(f"Error generating final response: {str(e)}")
        traceback.print_exc()
    
    # Default response if something goes wrong
    return "Your request has been processed successfully. The new item has been created according to your specifications. Is there anything else you'd like to do with it? You can make additional changes, create another item, or proceed to deployment."


def generate_followup_question(workflow: str, workflow_data: dict) -> str:
    """
    Enhanced follow-up question generator with better context awareness.
    """
    # First check if we have a conversation-directed followup already in the workflow data
    if "CONVERSATION" in workflow_data and workflow_data["CONVERSATION"]:
        return workflow_data["CONVERSATION"]
    
    # Find which field is missing
    missing_field = get_next_missing_field(workflow, workflow_data)
    
    # If using LLM for follow-ups, prepare the prompt
    schema_info = SCHEMA_INFO.get(workflow, "")
    prompt = followup_prompt.format(
        workflow_type="Clone-and-Modify" if workflow == "1" else "Create Entity",
        schema_info=schema_info,
        partial_json=json.dumps(workflow_data, indent=2)
    )
    
    # Try to get an AI-generated follow-up
    try:
        followup = invoke_bedrock_model(prompt=prompt, max_tokens=150, temperature=0.7)
        cleaned_followup = followup.strip()
        
        # Validate that the followup is a question
        if cleaned_followup and "?" in cleaned_followup:
            return cleaned_followup
    except Exception as e:
        print(f"Error generating AI followup: {str(e)}")
    
    # Fallback to more conversational pre-defined questions if AI generation fails
    if workflow == "1":  # Clone-and-modify existing offer
        if missing_field == "EXISTING_OFFER_NAME":
            return "Which existing offer would you like to use as a base for your new offer?"
        elif missing_field == "NEW_OFFER_NAME":
            existing = workflow_data.get("EXISTING_OFFER_NAME", "")
            return f"Great! Now, what would you like to name this new offer that's based on '{existing}'?"
        elif missing_field == "CATEGORY":
            new_name = workflow_data.get("NEW_OFFER_NAME", "")
            return f"Under which category should I place the new '{new_name}' offer?"
        elif missing_field == "ENTITIES_TO_MODIFY":
            existing = workflow_data.get("EXISTING_OFFER_NAME", "")
            # Get available entities for this offer
            available_entities = get_available_entities(existing)
            entity_list = format_entity_list(available_entities)
            return f"Which components of the '{existing}' offer would you like to modify? Here are the available components:\n\n{entity_list}"
        elif missing_field == "CHANGES":
            entities = workflow_data.get("ENTITIES_TO_MODIFY", "")
            return f"What specific changes would you like to make to the {entities}? For example, you can say things like 'limit to 2 MSISDNs' or 'increase recurring charge by 10%'."
    elif workflow == "2":  # Create new entity
        if missing_field == "ENTITY_TYPE":
            return "What type of entity would you like to create? You can choose from Package, Bundle, Promotion, Component, or Component Group."
        elif missing_field == "ENTITY_SUBTYPE":
            entity_type = workflow_data.get("ENTITY_TYPE", "entity")
            return f"What template or subtype should this {entity_type} use? You can say 'show me options' to see available templates."
        elif missing_field == "CATEGORY":
            entity_type = workflow_data.get("ENTITY_TYPE", "entity")
            subtype = workflow_data.get("ENTITY_SUBTYPE", "")
            return f"Which category should this new {entity_type} of type {subtype} be placed in? You can say 'show me options' to see available categories."
        elif missing_field == "BASE_ON_EXISTING":
            entity_type = workflow_data.get("ENTITY_TYPE", "entity")
            return f"Would you like to base this on an existing {entity_type} or create it from scratch? You can also say 'browse' to see available options."
        elif missing_field == "NEW_ENTITY_NAME":
            entity_type = workflow_data.get("ENTITY_TYPE", "entity")
            return f"What would you like to name this new {entity_type}?"
        elif missing_field == "CONFIGURATION_CHANGES":
            entity_type = workflow_data.get("ENTITY_TYPE", "entity")
            entity_name = workflow_data.get("NEW_ENTITY_NAME", "")
            return f"What specific configurations would you like for the {entity_name} {entity_type}? You can specify details like validity period, recurring charges, or any other parameters."
    
    # Default follow-up if something goes wrong
    return "Could you provide more details about what you'd like to do next?"


def get_next_missing_field(workflow: str, data: dict) -> str:
    """
    Enhanced function to get the next required field with better context awareness.
    """
    if workflow not in WORKFLOW_SCHEMAS:
        return ""
        
    schema = WORKFLOW_SCHEMAS[workflow]
    
    # Special case handling
    if workflow == "1":  # Clone-modify workflow
        # Custom ordering for clone-modify to ensure natural conversation flow
        field_order = [
            "EXISTING_OFFER_NAME", 
            "NEW_OFFER_NAME", 
            "CATEGORY", 
            "ENTITIES_TO_MODIFY", 
            "CHANGES"
        ]
        
        for field in field_order:
            if field not in data or data[field] == "NONE" or not data[field]:
                return field
                
    elif workflow == "2":  # Create entity workflow
        # If we're in a browsing state, no missing field to prompt for yet
        if "BROWSING_STATE" in data:
            return ""
            
        # If waiting for new name input
        if "NEW_NAME_NEEDED" in data and data["NEW_NAME_NEEDED"]:
            return "NEW_ENTITY_NAME"
        
        # Custom ordering for create entity to ensure natural conversation flow
        field_order = [
            "ENTITY_TYPE",
            "ENTITY_SUBTYPE",
            "CATEGORY",
            "BASE_ON_EXISTING",
            "NEW_ENTITY_NAME",
            "CONFIGURATION_CHANGES"
        ]
        
        for field in field_order:
            if field not in data or data[field] == "NONE" or not data[field]:
                return field
                
        # Special case for BASE_ON_EXISTING = YES
        if data.get("BASE_ON_EXISTING") == "YES":
            if "REFERENCE_ENTITY" not in data or not data["REFERENCE_ENTITY"]:
                return "REFERENCE_ENTITY"
            if "ENTITIES_TO_MODIFY" not in data or data["ENTITIES_TO_MODIFY"] == "NONE":
                return "ENTITIES_TO_MODIFY"
            if "CHANGES" not in data or data["CHANGES"] == "NONE":
                return "CHANGES"
    
    # Generic case - first required field that's missing
    for field in schema["required_fields"]:
        if field not in data or data[field] == "NONE" or not data[field]:
            return field
            
    return ""


def analyze_changes_and_set_action(workflow_data: dict) -> None:
    """
    Analyzes the requested changes and sets the appropriate action and override types.
    """
    changes = workflow_data.get("CHANGES", "")
    
    # Default to modify
    action = "modify"
    
    # Check for action indicators
    if re.search(r"\b(add|include|insert)\b", changes, re.IGNORECASE):
        action = "add"
    elif re.search(r"\b(remove|delete|exclude)\b", changes, re.IGNORECASE):
        action = "remove"
    elif re.search(r"\b(clone|copy|duplicate)\b", changes, re.IGNORECASE):
        action = "clone"
    
    workflow_data["ACTION"] = action
    
    # Analyze override types
    override_types = {}
    
    # Check for cardinality changes
    if re.search(r"\b(limit|maximum|minimum|max|min)\b.*?\b(\d+)\b", changes, re.IGNORECASE):
        override_types["cardinality"] = "cardinality"
    
    # Check for value changes
    if re.search(r"\b(increase|decrease|change|adjust|set)\b.*?\b(charge|price|cost|value|amount)\b", changes, re.IGNORECASE):
        override_types["value"] = "value"
    
    # Check for configuration changes
    if re.search(r"\b(data|allowance|gb|mb|gigabyte|megabyte|validity|period|days|months)\b", changes, re.IGNORECASE):
        override_types["config"] = "config"
    
    workflow_data["OVERRIDE_TYPES"] = override_types


def generate_confirmation_message(workflow_data: dict) -> str:
    """
    Generates a confirmation message showing all the captured details.
    """
    existing_offer = workflow_data.get("EXISTING_OFFER_NAME", "Unknown")
    new_offer = workflow_data.get("NEW_OFFER_NAME", "New Offer")
    category = workflow_data.get("CATEGORY", "Default Category")
    entities = workflow_data.get("ENTITIES_TO_MODIFY", "No components")
    changes = workflow_data.get("CHANGES", "No changes")
    
    # Format the changes for better readability
    formatted_changes = ""
    for change in changes.split(';'):
        change = change.strip()
        if change:
            formatted_changes += f"  • {change}\n"
    
    if not formatted_changes:
        formatted_changes = "  • No specific changes\n"
    
    return (
        f"**Offer Clone Summary**\n\n"
        f"I'll create a new offer with these details:\n\n"
        f"• Base offer: **{existing_offer}**\n"
        f"• New name: **{new_offer}**\n"
        f"• Category: **{category}**\n"
        f"• Components to modify: **{entities}**\n"
        f"• Changes to apply:\n{formatted_changes}\n"
        f"Is this correct? Or would you like to make any changes?"
    )

