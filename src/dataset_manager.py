"""
Dataset Manager for AI Email Response System.

Manages email-response pair datasets including loading, validation,
and statistical analysis. Supports synthetic data generation and
dataset quality assessment.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
from datetime import datetime

from src.utils import get_logger, load_json, save_json


# Initialize logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models (to be extended by task 2.1)
# ---------------------------------------------------------------------------

@dataclass
class EmailMetadata:
    """Metadata for email categorization and analysis."""
    subject: str
    formality_level: str  # "formal", "semi-formal", "casual"
    email_type: str       # "customer_support", "technical", "professional"
    subject_category: str # "inquiry", "complaint", "request", "follow-up", "announcement"
    sender_role: str      # "customer", "colleague", "manager", "vendor", "unknown"
    email_length_category: Optional[str] = None  # Computed: "short", "medium", "long"


@dataclass
class EmailPair:
    """Single email-response pair with metadata."""
    id: str
    incoming_email: str
    response: str
    metadata: EmailMetadata


# ---------------------------------------------------------------------------
# Dataset Manager Class
# ---------------------------------------------------------------------------

class DatasetManager:
    """
    Manages email dataset lifecycle including loading, validation,
    and statistical analysis.
    """
    
    def __init__(self):
        """Initialize the DatasetManager."""
        self.email_pairs: List[EmailPair] = []
        logger.info("DatasetManager initialized")
    
    # -----------------------------------------------------------------------
    # Synthetic Data Generation
    # -----------------------------------------------------------------------
    
    def generate_synthetic_dataset(self, size: int = 100, filepath: Optional[str] = None) -> List[EmailPair]:
        """
        Generate synthetic email-response pairs using structured templates.
        
        Distribution:
        - 30% Customer Support (product issues, refunds, feature questions)
        - 40% Professional Correspondence (meetings, projects, feedback)
        - 30% Technical Inquiries (bugs, integrations, documentation)
        
        Formality levels: formal, semi-formal, casual
        
        Args:
            size: Number of email pairs to generate (default: 100)
            filepath: Optional path to save generated dataset
            
        Returns:
            List of generated EmailPair objects
        """
        logger.info(f"Generating synthetic dataset with {size} pairs")
        
        email_pairs = []
        
        # Define target distribution
        customer_support_count = int(size * 0.30)
        professional_count = int(size * 0.40)
        technical_count = size - customer_support_count - professional_count
        
        # Generate each category
        email_pairs.extend(self._generate_customer_support_emails(customer_support_count))
        email_pairs.extend(self._generate_professional_emails(professional_count))
        email_pairs.extend(self._generate_technical_emails(technical_count))
        
        # Shuffle to mix categories
        random.shuffle(email_pairs)
        
        # Assign sequential IDs
        for idx, pair in enumerate(email_pairs, 1):
            pair.id = f"email_{idx:03d}"
        
        self.email_pairs = email_pairs
        
        # Save if filepath provided
        if filepath:
            self.save_dataset(filepath)
        
        logger.info(f"Generated {len(email_pairs)} synthetic email pairs")
        return email_pairs
    
    def _generate_customer_support_emails(self, count: int) -> List[EmailPair]:
        """Generate customer support email pairs."""
        templates = [
            {
                "incoming": "Hi there,\n\nI'm having trouble {issue}. I've tried {attempt} but it's still not working. Can you please help me resolve this as soon as possible?\n\nThanks",
                "response": "Thank you for reaching out to us. I understand you're experiencing issues with {issue_short}. I apologize for the inconvenience.\n\n{solution}\n\nIf you continue to experience problems, please don't hesitate to contact us again. We're here to help!\n\nBest regards",
                "formality": "semi-formal",
                "category": "complaint"
            },
            {
                "incoming": "Dear Support Team,\n\nI would like to inquire about {question}. Could you please provide me with detailed information regarding this matter? I need this information for {reason}.\n\nThank you for your assistance.",
                "response": "Dear Customer,\n\nThank you for your inquiry regarding {question_short}. I'm happy to provide you with the information you need.\n\n{answer}\n\nPlease let me know if you have any additional questions or require further clarification.\n\nKind regards",
                "formality": "formal",
                "category": "inquiry"
            },
            {
                "incoming": "Hey,\n\nJust wanted to check if you can {request}? I need this done by {timeframe} if possible. Let me know!\n\nCheers",
                "response": "Hi there!\n\nAbsolutely, I can help you with that. {confirmation}\n\n{details}\n\nLet me know if there's anything else you need!\n\nCheers",
                "formality": "casual",
                "category": "request"
            },
        ]
        
        variables = {
            "issue": [
                "logging into my account", "resetting my password",  
                "accessing the dashboard", "loading the application",
                "processing my payment", "downloading my files"
            ],
            "attempt": [
                "clearing my browser cache", "using a different browser",
                "restarting my device", "following the troubleshooting guide"
            ],
            "issue_short": [
                "login", "password reset", "dashboard access", 
                "application loading", "payment processing", "file downloads"
            ],
            "solution": [
                "I've reset your credentials and you should now be able to access your account. Please try logging in again with your email address.",
                "I've cleared the cache on our end and restarted the service. The issue should now be resolved. Please refresh your page and try again.",
                "I've checked your account and found the issue. I've applied a fix that should resolve this immediately. Please verify on your end.",
            ],
            "question": [
                "the refund policy for annual subscriptions",
                "how to upgrade my current plan to the premium tier",
                "the data retention policy for deleted files",
                "whether your service integrates with third-party applications",
                "the availability of custom enterprise solutions"
            ],
            "question_short": [
                "our refund policy", "plan upgrades", "data retention", 
                "third-party integrations", "enterprise solutions"
            ],
            "reason": [
                "our company's compliance requirements",
                "planning our budget for the next quarter",
                "evaluating your service for our organization"
            ],
            "answer": [
                "Our refund policy allows for full refunds within 30 days of purchase. After that period, we prorate refunds based on unused time. You can request a refund through your account settings or by contacting our billing team.",
                "You can upgrade your plan at any time through your account dashboard. The upgrade takes effect immediately, and you'll only be charged the prorated difference for the current billing period.",
                "We offer extensive integration options through our REST API and native connectors. Please visit our integrations page for a complete list of supported platforms, or contact our integration team for custom solutions.",
            ],
            "request": [
                "expedite my order", "update my billing information",
                "cancel my subscription", "extend my trial period",
                "resend my invoice"
            ],
            "timeframe": [
                "tomorrow", "end of the day", "this week", "the end of the month"
            ],
            "confirmation": [
                "I've processed your request and it's all set.",
                "I've taken care of this for you.",
                "Consider it done!"
            ],
            "details": [
                "The changes will take effect within the next 24 hours. You'll receive a confirmation email once everything is updated.",
                "I've sent you an email with all the details. Check your inbox in the next few minutes.",
                "Everything has been updated on our end. You should see the changes reflected in your account immediately.",
            ],
        }
        
        pairs = []
        for i in range(count):
            template = random.choice(templates)
            
            # Fill template with variables
            incoming = template["incoming"]
            response = template["response"]
            
            for key in variables:
                if "{" + key + "}" in incoming or "{" + key + "}" in response:
                    value = random.choice(variables[key])
                    incoming = incoming.replace("{" + key + "}", value)
                    response = response.replace("{" + key + "}", value)
            
            metadata = EmailMetadata(
                subject=f"Support Request #{1000 + i}",
                formality_level=template["formality"],
                email_type="customer_support",
                subject_category=template["category"],
                sender_role="customer"
            )
            
            pairs.append(EmailPair(
                id="", incoming_email=incoming, 
                response=response, metadata=metadata
            ))
        
        return pairs
    
    def _generate_professional_emails(self, count: int) -> List[EmailPair]:
        """Generate professional correspondence email pairs."""
        templates = [
            {
                "incoming": "Good morning {name},\n\nI hope this email finds you well. I wanted to reach out regarding {topic}. Would you be available for a brief meeting {timeframe} to discuss this further?\n\nI look forward to hearing from you.\n\nBest regards",
                "response": "Good morning,\n\nThank you for reaching out. I would be happy to discuss {topic_short} with you. {availability}\n\n{additional_info}\n\nPlease let me know what works best for your schedule.\n\nBest regards",
                "formality": "formal",
                "category": "request"
            },
            {
                "incoming": "Hi {name},\n\nQuick question about {topic}. Can you {ask}? I need this for {reason}.\n\nThanks!",
                "response": "Hi there,\n\n{answer}\n\n{details} Let me know if you need anything else!\n\nCheers",
                "formality": "casual",
                "category": "inquiry"
            },
            {
                "incoming": "Dear {name},\n\nI am writing to follow up on our previous discussion about {topic}. {question}\n\nI would appreciate your feedback on this matter at your earliest convenience.\n\nKind regards",
                "response": "Dear colleague,\n\nThank you for following up. Regarding {topic_short}, {response_content}\n\n{next_steps}\n\nPlease don't hesitate to reach out if you have further questions.\n\nBest regards",
                "formality": "formal",
                "category": "follow-up"
            },
        ]
        
        variables = {
            "name": ["Team", "there", "Sarah", "John", "Alex"],
            "topic": [
                "the Q4 project timeline and deliverables",
                "the upcoming team workshop scheduled for next month",
                "potential collaboration opportunities between our teams",
                "the budget allocation for the new initiative",
                "the proposed changes to our workflow process"
            ],
            "topic_short": [
                "the Q4 timeline", "the workshop", "collaboration opportunities",
                "budget allocation", "workflow changes"
            ],
            "timeframe": [
                "this week", "next Monday or Tuesday", "sometime in the next few days",
                "before the end of the month"
            ],
            "availability": [
                "I'm available this Thursday at 2pm or Friday at 10am.",
                "I have slots open on Tuesday afternoon or Wednesday morning.",
                "I can make time on Monday at 3pm or Thursday at 11am."
            ],
            "additional_info": [
                "I've also looped in our project manager who can provide additional context.",
                "I'll send over the relevant documents before our meeting.",
                "I've reviewed the materials you shared and have some thoughts to discuss."
            ],
            "ask": [
                "send me the latest report", "clarify the deadline",
                "review the attached document", "confirm the meeting time"
            ],
            "reason": [
                "the presentation tomorrow", "my records", "planning purposes",
                "the client meeting"
            ],
            "answer": [
                "Sure thing! I've attached the latest report to this email.",
                "The deadline is next Friday, the 15th. Let me know if you need an extension.",
                "I've reviewed the document and it looks good. Just a few minor suggestions in the comments.",
            ],
            "details": [
                "Everything should be in order now.",
                "I've copied the team so everyone is on the same page.",
                "I've updated the shared folder with the latest version.",
            ],
            "question": [
                "Could you provide an update on the current status?",
                "Have there been any developments I should be aware of?",
                "What are the next steps we should take?"
            ],
            "response_content": [
                "I've made significant progress and expect to complete the work by next week.",
                "we've encountered a few challenges but have identified potential solutions.",
                "the team has completed the initial phase and is ready to move forward.",
            ],
            "next_steps": [
                "I'll send you a detailed update by end of week with a revised timeline.",
                "Let's schedule a quick call to discuss the details and align on priorities.",
                "I'll circulate a summary document for review and approval.",
            ],
        }
        
        pairs = []
        for i in range(count):
            template = random.choice(templates)
            
            incoming = template["incoming"]
            response = template["response"]
            
            for key in variables:
                if "{" + key + "}" in incoming or "{" + key + "}" in response:
                    value = random.choice(variables[key])
                    incoming = incoming.replace("{" + key + "}", value)
                    response = response.replace("{" + key + "}", value)
            
            sender_roles = ["colleague", "manager", "colleague", "vendor"]
            
            metadata = EmailMetadata(
                subject=f"RE: {random.choice(['Project Update', 'Meeting Request', 'Collaboration', 'Follow-up', 'Discussion'])}",
                formality_level=template["formality"],
                email_type="professional",
                subject_category=template["category"],
                sender_role=random.choice(sender_roles)
            )
            
            pairs.append(EmailPair(
                id="", incoming_email=incoming,
                response=response, metadata=metadata
            ))
        
        return pairs
    
    def _generate_technical_emails(self, count: int) -> List[EmailPair]:
        """Generate technical inquiry email pairs."""
        templates = [
            {
                "incoming": "Hello,\n\nI'm experiencing a bug where {bug_description}. This happens when {reproduction_steps}. I'm using {environment}.\n\nError message: {error}\n\nCan you help me troubleshoot this?\n\nThanks",
                "response": "Hi there,\n\nThank you for reporting this issue. {acknowledgment}\n\n{solution}\n\n{additional_steps}\n\nPlease let me know if this resolves the issue or if you need further assistance.\n\nBest regards",
                "formality": "semi-formal",
                "category": "complaint"
            },
            {
                "incoming": "Hi Support,\n\nI'm trying to integrate your API with {platform}. I need to know {technical_question}. Could you provide documentation or guidance on this?\n\nThank you",
                "response": "Hello,\n\nThank you for your interest in integrating with our API. {intro}\n\n{technical_answer}\n\n{resources}\n\nIf you have any additional questions, feel free to reach out to our developer support team.\n\nRegards",
                "formality": "formal",
                "category": "inquiry"
            },
            {
                "incoming": "Hey team,\n\n{informal_question} I checked the docs but couldn't find info on this. Can someone point me in the right direction?\n\nCheers",
                "response": "Hey!\n\n{casual_answer}\n\n{helpful_tip}\n\nHope that helps! Let me know if you have more questions.\n\nCheers",
                "formality": "casual",
                "category": "inquiry"
            },
        ]
        
        variables = {
            "bug_description": [
                "the application crashes on startup",
                "data is not being saved correctly",
                "the API returns a 500 error",
                "images are not loading properly",
                "the search function returns no results"
            ],
            "reproduction_steps": [
                "I click the submit button after filling out the form",
                "I try to upload a file larger than 10MB",
                "I make multiple requests in quick succession",
                "I access the feature on mobile devices"
            ],
            "environment": [
                "version 2.4.1 on Windows 10 with Chrome",
                "the latest version on macOS Monterey",
                "version 2.3.0 on Ubuntu 20.04",
                "the web app on iOS Safari"
            ],
            "error": [
                "'NullPointerException at line 245'",
                "'Connection timeout after 30 seconds'",
                "'Invalid token provided'",
                "'Maximum upload size exceeded'"
            ],
            "acknowledgment": [
                "I've reviewed the error and identified the root cause.",
                "This is a known issue that we're actively working to resolve.",
                "I can see what's causing this problem on your account.",
            ],
            "solution": [
                "The issue was caused by a recent update. I've rolled back the changes on your account and the problem should now be fixed.",
                "This error occurs when the session expires. Please try logging out completely and logging back in, which should reset your session.",
                "I've identified a compatibility issue with your browser version. Please update to the latest version or try using Chrome/Firefox as a workaround.",
            ],
            "additional_steps": [
                "I've also submitted a bug report to our engineering team to ensure this doesn't happen again.",
                "In the meantime, I recommend using the alternative method described in our troubleshooting guide.",
                "If the problem persists, please send me your log files so I can investigate further.",
            ],
            "platform": [
                "our React application", "a Node.js backend", "a Python Flask app",
                "our mobile app built with React Native", "a WordPress site"
            ],
            "technical_question": [
                "how to implement OAuth2 authentication",
                "what the rate limits are for API requests",
                "how to handle webhook retries",
                "whether your API supports batch operations",
                "how to paginate through large result sets"
            ],
            "intro": [
                "We offer comprehensive API documentation to help with integration.",
                "I'd be happy to help you get started with the integration.",
                "Integration with your platform is definitely supported.",
            ],
            "technical_answer": [
                "For OAuth2 authentication, you'll need to register your application in our developer portal to obtain client credentials. Then implement the authorization code flow as described in our OAuth guide.",
                "Our API has a rate limit of 1000 requests per hour per API key. For higher limits, please consider our enterprise plan. Rate limit headers are included in every response.",
                "For pagination, use the 'page' and 'limit' query parameters. The response includes 'total_pages' and 'current_page' fields. Maximum page size is 100 items.",
            ],
            "resources": [
                "You can find detailed examples in our GitHub repository: github.com/ourcompany/api-examples",
                "I've also attached a Postman collection that demonstrates common API workflows.",
                "Our developer community forum has several threads discussing integration patterns that might be helpful.",
            ],
            "informal_question": [
                "How do I configure the cache settings?",
                "What's the difference between the v1 and v2 endpoints?",
                "Can I use webhooks for real-time updates?",
                "Is there a sandbox environment for testing?"
            ],
            "casual_answer": [
                "Yep! Cache settings are in the config.yaml file under the 'cache' section. You can set TTL and max size there.",
                "V2 endpoints use JSON exclusively and have better error handling. V1 is still supported but deprecated. Definitely use v2 for new projects!",
                "Absolutely! You can set up webhooks in your account settings. Just add your endpoint URL and select which events you want to receive.",
            ],
            "helpful_tip": [
                "Pro tip: use the debug mode flag to see detailed request/response logs.",
                "Also, make sure you're using the latest SDK version - it has some nice quality-of-life improvements.",
                "The changelog has examples for most common use cases if you need more details.",
            ],
        }
        
        pairs = []
        for i in range(count):
            template = random.choice(templates)
            
            incoming = template["incoming"]
            response = template["response"]
            
            for key in variables:
                if "{" + key + "}" in incoming or "{" + key + "}" in response:
                    value = random.choice(variables[key])
                    incoming = incoming.replace("{" + key + "}", value)
                    response = response.replace("{" + key + "}", value)
            
            metadata = EmailMetadata(
                subject=f"Technical Issue: {random.choice(['API', 'Bug Report', 'Integration', 'Documentation', 'Feature'])}",
                formality_level=template["formality"],
                email_type="technical",
                subject_category=template["category"],
                sender_role=random.choice(["customer", "colleague", "vendor"])
            )
            
            pairs.append(EmailPair(
                id="", incoming_email=incoming,
                response=response, metadata=metadata
            ))
        
        return pairs
    
    def save_dataset(self, filepath: str) -> None:
        """
        Save the current dataset to a JSON file.
        
        Args:
            filepath: Path where to save the dataset
        """
        if not self.email_pairs:
            logger.warning("No email pairs to save")
            return
        
        # Prepare dataset structure
        dataset = {
            "dataset_version": "1.0",
            "generation_method": "synthetic_templates",
            "total_pairs": len(self.email_pairs),
            "created_at": datetime.now().isoformat(),
            "email_pairs": []
        }
        
        # Convert email pairs to dictionaries
        for pair in self.email_pairs:
            pair_dict = {
                "id": pair.id,
                "incoming_email": pair.incoming_email,
                "response": pair.response,
                "metadata": {
                    "subject": pair.metadata.subject,
                    "formality_level": pair.metadata.formality_level,
                    "email_type": pair.metadata.email_type,
                    "subject_category": pair.metadata.subject_category,
                    "sender_role": pair.metadata.sender_role
                }
            }
            dataset["email_pairs"].append(pair_dict)
        
        # Save to file
        save_json(dataset, filepath)
        logger.info(f"Dataset saved to {filepath} ({len(self.email_pairs)} pairs)")
    
    # -----------------------------------------------------------------------
    # Validation Methods
    # -----------------------------------------------------------------------
    
    def validate_email_pair(self, pair_data: Dict) -> Tuple[bool, str]:
        """
        Validate a single email pair for structure and content requirements.
        
        Validation rules:
        - Incoming email: 50-2000 characters (after stripping)
        - Response: 50-1500 characters (after stripping)
        - Both fields non-empty after stripping whitespace
        - Required fields present: id, incoming_email, response, metadata
        - Metadata fields present and valid enum values
        
        Args:
            pair_data: Dictionary containing email pair data
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passes, False otherwise
            - error_message: Empty string if valid, descriptive error if invalid
            
        Example:
            >>> manager = DatasetManager()
            >>> data = {"id": "email_001", "incoming_email": "...", ...}
            >>> is_valid, error = manager.validate_email_pair(data)
        """
        # Check required top-level fields
        required_fields = ["id", "incoming_email", "response", "metadata"]
        for field in required_fields:
            if field not in pair_data:
                error_msg = f"Missing required field: {field}"
                logger.warning(f"Validation failed for pair: {error_msg}")
                return False, error_msg
        
        pair_id = pair_data.get("id", "unknown")
        
        # Validate incoming email
        incoming_email = pair_data.get("incoming_email", "")
        if not isinstance(incoming_email, str):
            error_msg = f"incoming_email must be a string (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        incoming_email_stripped = incoming_email.strip()
        if not incoming_email_stripped:
            error_msg = f"incoming_email cannot be empty after stripping whitespace (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        incoming_length = len(incoming_email_stripped)
        if incoming_length < 50:
            error_msg = f"incoming_email too short: {incoming_length} chars (minimum 50) (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        if incoming_length > 2000:
            error_msg = f"incoming_email too long: {incoming_length} chars (maximum 2000) (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        # Validate response
        response = pair_data.get("response", "")
        if not isinstance(response, str):
            error_msg = f"response must be a string (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        response_stripped = response.strip()
        if not response_stripped:
            error_msg = f"response cannot be empty after stripping whitespace (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        response_length = len(response_stripped)
        if response_length < 50:
            error_msg = f"response too short: {response_length} chars (minimum 50) (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        if response_length > 1500:
            error_msg = f"response too long: {response_length} chars (maximum 1500) (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        # Validate metadata presence
        metadata = pair_data.get("metadata", {})
        if not isinstance(metadata, dict):
            error_msg = f"metadata must be a dictionary (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        # Validate metadata fields
        required_metadata_fields = [
            "subject", "formality_level", "email_type", 
            "subject_category", "sender_role"
        ]
        for field in required_metadata_fields:
            if field not in metadata:
                error_msg = f"Missing required metadata field: {field} (pair_id: {pair_id})"
                logger.warning(f"Validation failed: {error_msg}")
                return False, error_msg
        
        # Validate enum values
        valid_formality_levels = ["formal", "semi-formal", "casual"]
        if metadata["formality_level"] not in valid_formality_levels:
            error_msg = f"Invalid formality_level: {metadata['formality_level']} (must be one of {valid_formality_levels}) (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        valid_email_types = ["customer_support", "technical", "professional"]
        if metadata["email_type"] not in valid_email_types:
            error_msg = f"Invalid email_type: {metadata['email_type']} (must be one of {valid_email_types}) (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        valid_subject_categories = ["inquiry", "complaint", "request", "follow-up", "announcement"]
        if metadata["subject_category"] not in valid_subject_categories:
            error_msg = f"Invalid subject_category: {metadata['subject_category']} (must be one of {valid_subject_categories}) (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        valid_sender_roles = ["customer", "colleague", "manager", "vendor", "unknown"]
        if metadata["sender_role"] not in valid_sender_roles:
            error_msg = f"Invalid sender_role: {metadata['sender_role']} (must be one of {valid_sender_roles}) (pair_id: {pair_id})"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
        
        # All validations passed
        logger.debug(f"Validation passed for pair_id: {pair_id}")
        return True, ""
    
    # -----------------------------------------------------------------------
    # Dataset Loading
    # -----------------------------------------------------------------------
    
    def load_dataset(self, filepath: str) -> List[EmailPair]:
        """
        Load and validate email dataset from JSON file.
        
        Expected JSON format:
        {
            "dataset_version": "1.0",
            "email_pairs": [
                {
                    "id": "email_001",
                    "incoming_email": "...",
                    "response": "...",
                    "metadata": {
                        "subject": "...",
                        "formality_level": "...",
                        "email_type": "...",
                        "subject_category": "...",
                        "sender_role": "..."
                    }
                }
            ]
        }
        
        Invalid pairs are logged and skipped. The method returns only
        valid EmailPair objects.
        
        Args:
            filepath: Path to JSON dataset file
            
        Returns:
            List of valid EmailPair objects
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If no valid email pairs are found in the dataset
            
        Example:
            >>> manager = DatasetManager()
            >>> pairs = manager.load_dataset("data/email_dataset.json")
            >>> print(f"Loaded {len(pairs)} email pairs")
        """
        logger.info(f"Loading dataset from: {filepath}")
        
        # Load JSON data
        try:
            data = load_json(filepath)
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset file: {filepath} - {e}")
            raise
        
        # Check for email_pairs field
        if "email_pairs" not in data:
            error_msg = f"Dataset file missing 'email_pairs' field: {filepath}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        email_pairs_data = data["email_pairs"]
        if not isinstance(email_pairs_data, list):
            error_msg = f"'email_pairs' must be a list, got {type(email_pairs_data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Found {len(email_pairs_data)} email pairs in dataset")
        
        # Validate and convert each pair
        valid_pairs = []
        invalid_count = 0
        
        for idx, pair_data in enumerate(email_pairs_data):
            is_valid, error_msg = self.validate_email_pair(pair_data)
            
            if not is_valid:
                invalid_count += 1
                logger.error(
                    f"Invalid email pair at index {idx}: {error_msg}",
                    extra={"pair_index": idx, "error": error_msg}
                )
                continue
            
            # Convert to EmailPair object
            try:
                email_pair = self._dict_to_email_pair(pair_data)
                valid_pairs.append(email_pair)
            except Exception as e:
                invalid_count += 1
                logger.error(
                    f"Failed to convert pair at index {idx} to EmailPair object: {e}",
                    extra={"pair_index": idx, "error": str(e)}
                )
                continue
        
        # Check if we have any valid pairs
        if not valid_pairs:
            error_msg = f"No valid email pairs found in dataset: {filepath}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log summary
        logger.info(
            f"Dataset loaded successfully: {len(valid_pairs)} valid pairs, {invalid_count} invalid pairs skipped",
            extra={
                "valid_pairs": len(valid_pairs),
                "invalid_pairs": invalid_count,
                "total_pairs": len(email_pairs_data)
            }
        )
        
        # Store in instance
        self.email_pairs = valid_pairs
        
        return valid_pairs
    
    def _dict_to_email_pair(self, pair_data: Dict) -> EmailPair:
        """
        Convert dictionary to EmailPair object.
        
        Args:
            pair_data: Dictionary containing email pair data
            
        Returns:
            EmailPair object
        """
        metadata_data = pair_data["metadata"]
        
        # Compute length category
        incoming_length = len(pair_data["incoming_email"].strip())
        if incoming_length < 500:
            length_category = "short"
        elif incoming_length <= 1000:
            length_category = "medium"
        else:
            length_category = "long"
        
        metadata = EmailMetadata(
            subject=metadata_data["subject"],
            formality_level=metadata_data["formality_level"],
            email_type=metadata_data["email_type"],
            subject_category=metadata_data["subject_category"],
            sender_role=metadata_data["sender_role"],
            email_length_category=length_category
        )
        
        return EmailPair(
            id=pair_data["id"],
            incoming_email=pair_data["incoming_email"].strip(),
            response=pair_data["response"].strip(),
            metadata=metadata
        )
    
    # -----------------------------------------------------------------------
    # Dataset Statistics
    # -----------------------------------------------------------------------
    
    def get_dataset_statistics(self) -> Dict:
        """
        Compute and return comprehensive dataset statistics.
        
        Statistics include:
        - Total number of email pairs
        - Distribution of formality levels
        - Distribution of email types
        - Distribution of subject categories
        - Distribution of sender roles
        - Distribution of length categories (short <500, medium 500-1000, long >1000)
        - Character length statistics (min, max, mean, median)
        
        Returns:
            Dictionary containing dataset statistics
            
        Example:
            >>> manager = DatasetManager()
            >>> manager.load_dataset("data/email_dataset.json")
            >>> stats = manager.get_dataset_statistics()
            >>> print(f"Total pairs: {stats['total_pairs']}")
            >>> print(f"Formality distribution: {stats['formality_level_distribution']}")
        """
        if not self.email_pairs:
            logger.warning("No email pairs loaded, returning empty statistics")
            return {
                "total_pairs": 0,
                "error": "No email pairs loaded"
            }
        
        logger.info("Computing dataset statistics")
        
        # Initialize counters
        formality_levels = []
        email_types = []
        subject_categories = []
        sender_roles = []
        length_categories = []
        incoming_lengths = []
        response_lengths = []
        
        # Collect data
        for pair in self.email_pairs:
            formality_levels.append(pair.metadata.formality_level)
            email_types.append(pair.metadata.email_type)
            subject_categories.append(pair.metadata.subject_category)
            sender_roles.append(pair.metadata.sender_role)
            length_categories.append(pair.metadata.email_length_category)
            incoming_lengths.append(len(pair.incoming_email))
            response_lengths.append(len(pair.response))
        
        # Compute distributions
        def distribution_dict(values: List[str]) -> Dict[str, int]:
            """Convert list of values to distribution dictionary."""
            return dict(Counter(values))
        
        def distribution_percentages(values: List[str]) -> Dict[str, float]:
            """Convert list of values to percentage distribution."""
            counts = Counter(values)
            total = len(values)
            return {k: round(v / total * 100, 2) for k, v in counts.items()}
        
        # Compute length statistics
        def length_stats(lengths: List[int]) -> Dict[str, float]:
            """Compute length statistics."""
            sorted_lengths = sorted(lengths)
            n = len(sorted_lengths)
            return {
                "min": min(sorted_lengths),
                "max": max(sorted_lengths),
                "mean": round(sum(sorted_lengths) / n, 2),
                "median": sorted_lengths[n // 2] if n % 2 == 1 else 
                         round((sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2, 2)
            }
        
        # Build statistics dictionary
        statistics = {
            "total_pairs": len(self.email_pairs),
            "formality_level_distribution": {
                "counts": distribution_dict(formality_levels),
                "percentages": distribution_percentages(formality_levels)
            },
            "email_type_distribution": {
                "counts": distribution_dict(email_types),
                "percentages": distribution_percentages(email_types)
            },
            "subject_category_distribution": {
                "counts": distribution_dict(subject_categories),
                "percentages": distribution_percentages(subject_categories)
            },
            "sender_role_distribution": {
                "counts": distribution_dict(sender_roles),
                "percentages": distribution_percentages(sender_roles)
            },
            "length_category_distribution": {
                "counts": distribution_dict(length_categories),
                "percentages": distribution_percentages(length_categories)
            },
            "incoming_email_length_stats": length_stats(incoming_lengths),
            "response_length_stats": length_stats(response_lengths)
        }
        
        logger.info(
            f"Dataset statistics computed: {statistics['total_pairs']} pairs",
            extra={"total_pairs": statistics["total_pairs"]}
        )
        
        return statistics
    
    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------
    
    def get_email_pair_by_id(self, pair_id: str) -> Optional[EmailPair]:
        """
        Retrieve a specific email pair by ID.
        
        Args:
            pair_id: Email pair ID to retrieve
            
        Returns:
            EmailPair object if found, None otherwise
        """
        for pair in self.email_pairs:
            if pair.id == pair_id:
                return pair
        return None
    
    def filter_by_metadata(self, **filters) -> List[EmailPair]:
        """
        Filter email pairs by metadata attributes.
        
        Args:
            **filters: Keyword arguments for filtering (e.g., formality_level="formal")
            
        Returns:
            List of EmailPair objects matching the filters
            
        Example:
            >>> manager = DatasetManager()
            >>> manager.load_dataset("data/email_dataset.json")
            >>> formal_pairs = manager.filter_by_metadata(formality_level="formal")
            >>> support_pairs = manager.filter_by_metadata(email_type="customer_support")
        """
        filtered_pairs = self.email_pairs
        
        for key, value in filters.items():
            filtered_pairs = [
                pair for pair in filtered_pairs
                if hasattr(pair.metadata, key) and getattr(pair.metadata, key) == value
            ]
        
        logger.debug(f"Filtered {len(filtered_pairs)} pairs with filters: {filters}")
        return filtered_pairs
