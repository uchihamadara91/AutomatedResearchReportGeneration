import uuid
import os
from fastapi.responses import FileResponse
from research_and_analyst.utils.model_loader import ModelLoader
from research_and_analyst.workflows.report_generator_workflow import AutonomousReportGenerator
from research_and_analyst.logger import GLOBAL_LOGGER
from research_and_analyst.exception.custom_exception import ResearchAnalystException
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path
from fastapi.responses import FileResponse

_shared_memory = MemorySaver()

class ReportService:
    def __init__(self):
        self.llm = ModelLoader().load_llm()
        self.reporter = AutonomousReportGenerator(self.llm)
        self.reporter.memory = _shared_memory
        self.graph = self.reporter.build_graph()
        self.logger = GLOBAL_LOGGER.bind(module="ReportService")

    def start_report_generation(self, topic: str, max_analysts: int):
        """Trigger the autonomous report pipeline."""
        try:
            thread_id = str(uuid.uuid4())
            thread = {"configurable": {"thread_id": thread_id}}
            self.logger.info("Starting report pipeline", topic=topic, thread_id=thread_id)

            for _ in self.graph.stream({"topic": topic, "max_analysts": max_analysts}, thread, stream_mode="values"):
                pass

            return {"thread_id": thread_id, "message": "Pipeline initiated successfully."}
        except Exception as e:
            self.logger.error("Error initiating report generation", error=str(e))
            raise ResearchAnalystException("Failed to start report generation", e)

    def submit_feedback(self, thread_id: str, feedback: str):
        """Update human feedback in graph state."""
        try:
            thread = {"configurable": {"thread_id": thread_id}}
            self.graph.update_state(thread, {"human_analyst_feedback": feedback}, as_node="human_feedback")
            self.logger.info("Feedback updated", thread_id=thread_id)
            for _ in self.graph.stream(None, thread, stream_mode="values"):
                pass
            return {"message": "Feedback processed successfully"}
        except Exception as e:
            self.logger.error("Error updating feedback", error=str(e))
            raise ResearchAnalystException("Failed to update feedback", e)
        
    def get_report_status(self, thread_id: str):
        """Fetch latest state or final report."""
        try:
            thread = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(thread)
            final_report = state.values.get("final_report")
            topic = state.values.get("topic", "AI_Report") 

            if final_report:
                # now topic-based report folder name
                file_docx = self.reporter.save_report(final_report, topic, "docx")
                file_pdf = self.reporter.save_report(final_report, topic, "pdf")
                return {
                    "status": "completed",
                    "docx_path": file_docx,
                    "pdf_path": file_pdf,
                }
            return {"status": "in_progress"}
        except Exception as e:
            self.logger.error("Error fetching report status", error=str(e))
            raise ResearchAnalystException("Failed to fetch report status", e)

    @staticmethod
    def download_file(file_name: str):
        """
        Find a generated file by name anywhere under <project_root>/generated_report
        and stream it if found; else return a clean JSON error.
        """
        # currently you probably have:
        # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        # but because this file is inside research_and_analyst/api/services/
        # you must go up THREE levels, not two.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        report_dir = os.path.join(project_root, "generated_report")

        for root, _, files in os.walk(report_dir):
            if file_name in files:
                path = os.path.join(root, file_name)
                return FileResponse(path=path, filename=file_name, media_type="application/octet-stream")

        return {"error": f"File {file_name} not found under {report_dir}"}
