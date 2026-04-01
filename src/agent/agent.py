"""
Care Coordination LLM Agent
============================
Routes natural language queries to the right tool (Graph DB or Vector DB)
using OpenAI's function calling API.

Usage:
    from src.agent.agent import CareAgent
    agent = CareAgent()
    response = agent.chat("What diseases does Soledad have?")
"""

import json
from openai import OpenAI
from src.tools.graph_tools import GraphTools
from src.tools.vector_tools import VectorTools


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Care Coordination Healthcare Analyst AI. You help healthcare professionals 
analyze patient data stored in a Knowledge Graph (Neo4j) and a Vector Database (Qdrant).

You have access to tools that query these databases. When the user asks a question:
1. Pick the most appropriate tool(s) to answer it.
2. Call the tool with the correct parameters.
3. Interpret the results and give a clear, professional answer.

IMPORTANT GUIDELINES:
- Always use tools to get data. Never make up patient information.
- For patient queries, use partial name matching (e.g., user says "Soledad", pass "Soledad").
- When asked about risk or clustering, use Vector DB tools (get_financial_risk, get_care_coordination).
- When asked about specific patient details, diseases, providers, or costs, use Graph DB tools.
- When asked to find similar patients, use get_similar_patients.
- For general graph statistics, use get_graph_stats.
- If no predefined tool fits, use run_cypher_query to write a custom Cypher query.
- Format monetary values with $ and commas (e.g., $1,234.56).
- Be concise but thorough in your responses.

EXAMPLES OF TOOL SELECTION:
- "Tell me about Soledad" → get_patient_info
- "Who treated Giovanni?" → get_patient_providers
- "How much did Soledad's care cost?" → get_patient_cost
- "What are the most common diseases?" → get_top_diseases
- "Which diseases occur together?" → get_comorbidities
- "Is Soledad high risk?" → get_financial_risk
- "Find patients like Giovanni" → get_similar_patients
- "How is Soledad's care coordination?" → get_care_coordination
- "Show me payer breakdown" → get_payer_breakdown
- "How many female patients do we have?" → run_cypher_query
"""


# ═══════════════════════════════════════════════════════════════
# TOOL DEFINITIONS (OpenAI function calling format)
# ═══════════════════════════════════════════════════════════════

TOOLS = [
    # ── Graph DB Tools ──────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_patient_info",
            "description": "Get detailed information about a patient including demographics, conditions, encounter count, and provider count. Use for questions like 'tell me about [patient]', 'what diseases does [patient] have', 'patient details'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full or partial name of the patient to search for."
                    }
                },
                "required": ["patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_providers",
            "description": "Find all healthcare providers who have treated a specific patient, including visit counts and organizations. Use for 'who treated [patient]', 'which doctors saw [patient]'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full or partial name of the patient."
                    }
                },
                "required": ["patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_cost",
            "description": "Get total healthcare cost and claims breakdown by payer for a patient. Use for 'how much did [patient] cost', 'what is [patient]'s total spend'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full or partial name of the patient."
                    }
                },
                "required": ["patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_diseases",
            "description": "Find the most common diseases across all patients. Use for 'most common diseases', 'top conditions', 'prevalent diseases'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top diseases to return. Default is 10.",
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_comorbidities",
            "description": "Find diseases that frequently co-occur in the same patients. Can filter by a specific disease or show global pairs. Use for 'which diseases occur together', 'comorbidities of asthma'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {
                        "type": "string",
                        "description": "Optional: filter co-occurrences for a specific disease. If omitted, returns top global pairs."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top pairs to return. Default is 10.",
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_provider_network",
            "description": "Find provider pairs that share the most patients (care coordination network). Use for 'which doctors share patients', 'provider connections'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_name": {
                        "type": "string",
                        "description": "Optional: find connections for a specific provider."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top pairs to return. Default is 10.",
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_payer_breakdown",
            "description": "Get total cost and claim count breakdown by insurance payer (Medicare, Medicaid, etc.). Use for 'payer breakdown', 'insurance costs', 'how much did Medicare pay'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_graph_stats",
            "description": "Get overall statistics of the knowledge graph including node counts and relationship counts. Use for 'how many patients', 'graph statistics', 'database overview'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_cypher_query",
            "description": "Execute a custom Cypher query against the Neo4j knowledge graph. Use ONLY when no other tool fits. You must write a valid Cypher query. Schema: Patient(id,name,gender,birth_date,city,state), Disease(name), Encounter(id,reason,period_start,period_end), Provider(id,name,type,service_provider), Claim(id,amount), Payer(name). Relationships: HAS_ENCOUNTER, WITH_PROVIDER, DIAGNOSED, HAS_CONDITION, GENERATED_CLAIM, FOR_PATIENT, FULFILLED_BY.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cypher": {
                        "type": "string",
                        "description": "A valid Cypher query string. Only read operations (MATCH/RETURN) are allowed."
                    }
                },
                "required": ["cypher"]
            }
        }
    },
    # ── Vector DB Tools ─────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_financial_risk",
            "description": "Classify a patient's financial risk level as 'High Risk' or 'Low Risk' using ML clustering on financial and claims data. Use for 'is [patient] high risk', 'financial risk of [patient]', 'risk level'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full or partial name of the patient."
                    }
                },
                "required": ["patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_care_coordination",
            "description": "Classify a patient's care coordination quality as 'Optimal Care' or 'Non-Optimal Care'. Analyzes provider diversity, visit patterns, and clinical data. Use for 'care quality of [patient]', 'is [patient] getting good care'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full or partial name of the patient."
                    }
                },
                "required": ["patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_patients",
            "description": "Find patients most similar to a given patient based on combined clinical, financial, and behavioural embeddings. Use for 'find patients like [patient]', 'who is similar to [patient]'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full or partial name of the patient."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar patients to return. Default is 5.",
                        "default": 5
                    }
                },
                "required": ["patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster_summary",
            "description": "Get overall summary of patient clustering across financial risk and care coordination. Use for 'overall clustering', 'population risk overview', 'cluster distribution'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]


# ═══════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════

class CareAgent:
    """LLM Agent that routes healthcare queries to Graph DB and Vector DB tools."""

    def __init__(
        self,
        openai_api_key: str = None,
        model: str = "gpt-4o-mini",
        neo4j_uri: str = "neo4j://127.0.0.1:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "12345678",
    ):
        """Initialize the agent with OpenAI and tool connections.

        Args:
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-4o-mini).
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
        """
        # OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

        # Initialize tools
        self.graph_tools = GraphTools(neo4j_uri, neo4j_user, neo4j_password)

        try:
            self.vector_tools = VectorTools()
            self._vector_available = True
        except Exception as e:
            print(f"⚠️  Vector tools unavailable: {e}")
            self.vector_tools = None
            self._vector_available = False

        # Conversation history
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Map tool names to actual functions
        self.tool_map = {
            # Graph tools
            "get_patient_info": self.graph_tools.get_patient_info,
            "get_patient_providers": self.graph_tools.get_patient_providers,
            "get_patient_cost": self.graph_tools.get_patient_cost,
            "get_top_diseases": self.graph_tools.get_top_diseases,
            "get_comorbidities": self.graph_tools.get_comorbidities,
            "get_provider_network": self.graph_tools.get_provider_network,
            "get_payer_breakdown": self.graph_tools.get_payer_breakdown,
            "get_graph_stats": self.graph_tools.get_graph_stats,
            "run_cypher_query": self.graph_tools.run_cypher_query,
        }

        # Add vector tools if available
        if self._vector_available:
            self.tool_map.update({
                "get_financial_risk": self.vector_tools.get_financial_risk,
                "get_care_coordination": self.vector_tools.get_care_coordination,
                "get_similar_patients": self.vector_tools.get_similar_patients,
                "get_cluster_summary": self.vector_tools.get_cluster_summary,
            })

    def _get_available_tools(self) -> list:
        """Return tool definitions, excluding vector tools if unavailable."""
        if self._vector_available:
            return TOOLS
        # Filter out vector tools
        vector_tool_names = {"get_financial_risk", "get_care_coordination",
                            "get_similar_patients", "get_cluster_summary"}
        return [t for t in TOOLS if t["function"]["name"] not in vector_tool_names]

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool by name and return JSON result."""
        func = self.tool_map.get(name)
        if not func:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = func(**arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    def chat(self, user_message: str) -> str:
        """Send a message to the agent and get a response.

        Args:
            user_message: Natural language question from the user.

        Returns:
            Agent's natural language response.
        """
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_message})

        # Call OpenAI with tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self._get_available_tools(),
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # Handle tool calls (may be multiple)
        while assistant_message.tool_calls:
            # Add assistant message with tool calls
            self.messages.append(assistant_message)

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"  🔧 Calling: {func_name}({arguments})")

                result = self._execute_tool(func_name, arguments)

                # Add tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Get next response (LLM interprets tool results)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self._get_available_tools(),
                tool_choice="auto",
            )
            assistant_message = response.choices[0].message

        # Final text response
        reply = assistant_message.content
        self.messages.append({"role": "assistant", "content": reply})

        return reply

    def reset(self):
        """Clear conversation history (start fresh)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def close(self):
        """Clean up connections."""
        self.graph_tools.close()
