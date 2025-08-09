"""
Rollback Automation - Enterprise Migration Safety & Recovery Platform
Automated rollback procedures and disaster recovery for enterprise website migrations

🎯 PORTFOLIO PROJECT: Demonstrates disaster recovery expertise and automation capabilities
Perfect for: DevOps engineers, technical project managers, enterprise architects

📄 DEMO/PORTFOLIO CODE: This is demonstration code showcasing automated recovery capabilities.
   Real implementations require comprehensive infrastructure integration and testing procedures.

🔗 Connect with the developer: https://www.linkedin.com/in/sspyrou/
🚀 AI-Enhanced Migration Solutions: https://verityai.co

Built by a technical marketing leader with proven experience in zero-downtime migrations
and automated recovery systems that protected enterprise revenue during critical deployments.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import shutil
import tempfile

import aiohttp
import pandas as pd
from jinja2 import Template


class RollbackTrigger(Enum):
    """Rollback trigger conditions"""
    TRAFFIC_DROP = "traffic_drop"
    RANKING_DROP = "ranking_drop"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    HEALTH_CHECK = "health_check"


class RollbackStatus(Enum):
    """Rollback execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class RollbackAction:
    """Single rollback action"""
    action_id: str
    action_type: str  # dns, redirect, content, config
    description: str
    priority: int = 1  # 1=highest, 10=lowest
    rollback_command: str = ""
    validation_command: str = ""
    estimated_duration: int = 60  # seconds
    dependencies: List[str] = field(default_factory=list)
    status: RollbackStatus = RollbackStatus.PENDING
    executed_at: Optional[datetime] = None
    execution_log: List[str] = field(default_factory=list)


@dataclass
class MigrationSnapshot:
    """Snapshot of pre-migration state"""
    snapshot_id: str
    created_at: datetime
    description: str
    dns_records: Dict[str, Any] = field(default_factory=dict)
    redirects: List[Dict[str, str]] = field(default_factory=list)
    sitemap_urls: List[str] = field(default_factory=list)
    key_metrics: Dict[str, float] = field(default_factory=dict)
    config_backups: Dict[str, str] = field(default_factory=dict)  # file -> backup_path
    database_backup: Optional[str] = None


@dataclass
class RollbackPlan:
    """Complete rollback execution plan"""
    plan_id: str
    migration_id: str
    created_at: datetime
    trigger_condition: RollbackTrigger
    trigger_details: str
    pre_migration_snapshot: MigrationSnapshot
    rollback_actions: List[RollbackAction]
    validation_checks: List[str]
    estimated_total_duration: int = 0
    max_acceptable_downtime: int = 300  # seconds


class RollbackAutomation:
    """Automated rollback system for SEO migrations"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_rollbacks: Dict[str, RollbackPlan] = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "monitoring": {
                "traffic_drop_threshold": 0.15,  # 15% drop triggers rollback
                "ranking_drop_threshold": 5,     # Average position drop
                "error_rate_threshold": 0.05,   # 5% error rate
                "monitoring_window": 3600,      # 1 hour monitoring window
                "check_interval": 300           # Check every 5 minutes
            },
            "rollback": {
                "max_parallel_actions": 3,
                "action_timeout": 600,
                "total_timeout": 3600,
                "auto_rollback_enabled": True,
                "require_confirmation": False,
                "backup_retention_days": 30
            },
            "notifications": {
                "email_alerts": True,
                "slack_webhooks": [],
                "sms_alerts": False,
                "dashboard_alerts": True
            },
            "safety": {
                "staging_test_required": True,
                "manual_approval_for_prod": True,
                "rollback_window": 86400,  # 24 hours
                "max_rollbacks_per_day": 3
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def create_migration_snapshot(
        self, 
        migration_id: str,
        description: str = ""
    ) -> MigrationSnapshot:
        """Create pre-migration snapshot"""
        self.logger.info(f"Creating migration snapshot for {migration_id}")
        
        snapshot = MigrationSnapshot(
            snapshot_id=f"snapshot_{migration_id}_{int(datetime.now().timestamp())}",
            created_at=datetime.now(),
            description=description or f"Pre-migration snapshot for {migration_id}"
        )
        
        # Capture DNS records (mock implementation)
        snapshot.dns_records = self._capture_dns_records()
        
        # Capture redirect rules
        snapshot.redirects = self._capture_redirect_rules()
        
        # Capture sitemap URLs
        snapshot.sitemap_urls = self._capture_sitemap_urls()
        
        # Capture key metrics baseline
        snapshot.key_metrics = self._capture_baseline_metrics()
        
        # Backup configuration files
        snapshot.config_backups = self._backup_config_files()
        
        # Create database backup
        snapshot.database_backup = self._create_database_backup()
        
        # Save snapshot
        self._save_snapshot(snapshot)
        
        self.logger.info(f"Snapshot created: {snapshot.snapshot_id}")
        return snapshot
    
    def _capture_dns_records(self) -> Dict[str, Any]:
        """Capture current DNS configuration"""
        # In practice, would query DNS provider API
        return {
            "a_records": [
                {"name": "www", "value": "192.168.1.1", "ttl": 300},
                {"name": "@", "value": "192.168.1.1", "ttl": 300}
            ],
            "cname_records": [
                {"name": "cdn", "value": "cdn.example.com", "ttl": 300}
            ],
            "mx_records": [
                {"name": "@", "value": "mail.example.com", "priority": 10}
            ]
        }
    
    def _capture_redirect_rules(self) -> List[Dict[str, str]]:
        """Capture current redirect configuration"""
        # In practice, would read from web server config or CDN
        return [
            {"from": "/old-page", "to": "/new-page", "type": "301"},
            {"from": "/legacy/*", "to": "/modern/$1", "type": "301"}
        ]
    
    def _capture_sitemap_urls(self) -> List[str]:
        """Capture current sitemap URLs"""
        # In practice, would parse sitemap.xml
        return [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/contact"
        ]
    
    def _capture_baseline_metrics(self) -> Dict[str, float]:
        """Capture baseline performance metrics"""
        # In practice, would query analytics APIs
        return {
            "organic_traffic_7d": 10000.0,
            "avg_ranking_position": 15.2,
            "core_web_vitals_lcp": 2.1,
            "error_rate": 0.002,
            "conversion_rate": 0.025
        }
    
    def _backup_config_files(self) -> Dict[str, str]:
        """Backup critical configuration files"""
        backup_dir = Path(tempfile.mkdtemp(prefix="migration_backup_"))
        backups = {}
        
        # Mock config files to backup
        config_files = [
            "/etc/nginx/nginx.conf",
            "/etc/apache2/apache2.conf", 
            "/var/www/html/.htaccess",
            "/app/config/app.yaml"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                backup_path = backup_dir / config_path.name
                try:
                    shutil.copy2(config_file, backup_path)
                    backups[config_file] = str(backup_path)
                    self.logger.info(f"Backed up {config_file} to {backup_path}")
                except Exception as e:
                    self.logger.error(f"Failed to backup {config_file}: {e}")
        
        return backups
    
    def _create_database_backup(self) -> Optional[str]:
        """Create database backup"""
        backup_path = f"/tmp/db_backup_{int(datetime.now().timestamp())}.sql"
        
        # Mock database backup
        try:
            # In practice, would run mysqldump, pg_dump, etc.
            with open(backup_path, 'w') as f:
                f.write("-- Database backup created at " + datetime.now().isoformat())
            
            self.logger.info(f"Database backup created: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create database backup: {e}")
            return None
    
    def _save_snapshot(self, snapshot: MigrationSnapshot):
        """Save snapshot to storage"""
        snapshot_dir = Path("snapshots")
        snapshot_dir.mkdir(exist_ok=True)
        
        snapshot_file = snapshot_dir / f"{snapshot.snapshot_id}.json"
        
        # Convert snapshot to JSON-serializable format
        snapshot_data = {
            "snapshot_id": snapshot.snapshot_id,
            "created_at": snapshot.created_at.isoformat(),
            "description": snapshot.description,
            "dns_records": snapshot.dns_records,
            "redirects": snapshot.redirects,
            "sitemap_urls": snapshot.sitemap_urls,
            "key_metrics": snapshot.key_metrics,
            "config_backups": snapshot.config_backups,
            "database_backup": snapshot.database_backup
        }
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        self.logger.info(f"Snapshot saved to {snapshot_file}")
    
    def create_rollback_plan(
        self,
        migration_id: str,
        snapshot: MigrationSnapshot,
        trigger: RollbackTrigger,
        trigger_details: str = ""
    ) -> RollbackPlan:
        """Create comprehensive rollback plan"""
        plan_id = f"rollback_{migration_id}_{int(datetime.now().timestamp())}"
        
        plan = RollbackPlan(
            plan_id=plan_id,
            migration_id=migration_id,
            created_at=datetime.now(),
            trigger_condition=trigger,
            trigger_details=trigger_details,
            pre_migration_snapshot=snapshot,
            rollback_actions=[]
        )
        
        # Generate rollback actions based on snapshot
        self._generate_rollback_actions(plan)
        
        # Calculate total estimated duration
        plan.estimated_total_duration = sum(
            action.estimated_duration for action in plan.rollback_actions
        )
        
        self.logger.info(f"Rollback plan created: {plan_id}")
        return plan
    
    def _generate_rollback_actions(self, plan: RollbackPlan):
        """Generate rollback actions based on snapshot"""
        actions = []
        
        # DNS rollback actions
        if plan.pre_migration_snapshot.dns_records:
            actions.append(RollbackAction(
                action_id="dns_restore",
                action_type="dns",
                description="Restore DNS records to pre-migration state",
                priority=1,
                rollback_command="restore_dns_records",
                validation_command="verify_dns_propagation",
                estimated_duration=300  # 5 minutes for DNS propagation
            ))
        
        # Redirect rollback actions
        if plan.pre_migration_snapshot.redirects:
            actions.append(RollbackAction(
                action_id="redirects_restore",
                action_type="redirect",
                description="Restore redirect rules",
                priority=2,
                rollback_command="restore_redirect_rules",
                validation_command="test_redirect_rules",
                estimated_duration=60,
                dependencies=["dns_restore"]
            ))
        
        # Configuration rollback actions
        if plan.pre_migration_snapshot.config_backups:
            actions.append(RollbackAction(
                action_id="config_restore",
                action_type="config",
                description="Restore configuration files",
                priority=2,
                rollback_command="restore_config_files",
                validation_command="validate_config_syntax",
                estimated_duration=120
            ))
        
        # Database rollback actions
        if plan.pre_migration_snapshot.database_backup:
            actions.append(RollbackAction(
                action_id="database_restore",
                action_type="database",
                description="Restore database to pre-migration state",
                priority=3,
                rollback_command="restore_database",
                validation_command="verify_database_integrity",
                estimated_duration=600,  # 10 minutes
                dependencies=["config_restore"]
            ))
        
        # Content/cache refresh actions
        actions.append(RollbackAction(
            action_id="cache_purge",
            action_type="cache",
            description="Purge CDN and application caches",
            priority=4,
            rollback_command="purge_all_caches",
            validation_command="verify_cache_purged",
            estimated_duration=180,
            dependencies=["redirects_restore", "config_restore"]
        ))
        
        # Search engine notification
        actions.append(RollbackAction(
            action_id="search_notify",
            action_type="seo",
            description="Notify search engines of rollback",
            priority=5,
            rollback_command="submit_sitemap_changes",
            validation_command="verify_sitemap_submitted",
            estimated_duration=60,
            dependencies=["cache_purge"]
        ))
        
        plan.rollback_actions = sorted(actions, key=lambda x: x.priority)
    
    async def execute_rollback(
        self, 
        plan: RollbackPlan,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute rollback plan"""
        if dry_run:
            self.logger.info(f"DRY RUN: Executing rollback plan {plan.plan_id}")
        else:
            self.logger.info(f"Executing rollback plan {plan.plan_id}")
            self.active_rollbacks[plan.plan_id] = plan
        
        results = {
            "plan_id": plan.plan_id,
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "actions": [],
            "overall_status": "in_progress"
        }
        
        # Send notification
        if not dry_run:
            await self._send_rollback_notification(
                "started", plan, "Rollback execution started"
            )
        
        try:
            # Execute actions in dependency order
            executed_actions = set()
            max_parallel = self.config["rollback"]["max_parallel_actions"]
            
            while len(executed_actions) < len(plan.rollback_actions):
                # Find ready actions (dependencies satisfied)
                ready_actions = [
                    action for action in plan.rollback_actions
                    if (action.action_id not in executed_actions and
                        all(dep in executed_actions for dep in action.dependencies))
                ]
                
                if not ready_actions:
                    break  # Circular dependencies or other issue
                
                # Execute up to max_parallel actions
                batch = ready_actions[:max_parallel]
                
                if dry_run:
                    # Simulate execution
                    for action in batch:
                        result = await self._simulate_action(action)
                        results["actions"].append(result)
                        executed_actions.add(action.action_id)
                else:
                    # Actually execute
                    tasks = [self._execute_action(action, plan) for action in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for action, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            result = {
                                "action_id": action.action_id,
                                "status": "failed",
                                "error": str(result),
                                "executed_at": datetime.now().isoformat()
                            }
                        
                        results["actions"].append(result)
                        executed_actions.add(action.action_id)
            
            # Determine overall status
            failed_actions = [r for r in results["actions"] if r["status"] == "failed"]
            if not failed_actions:
                results["overall_status"] = "completed"
            elif len(failed_actions) < len(results["actions"]):
                results["overall_status"] = "partially_completed"
            else:
                results["overall_status"] = "failed"
            
            results["completed_at"] = datetime.now().isoformat()
            
            # Send completion notification
            if not dry_run:
                await self._send_rollback_notification(
                    results["overall_status"], plan, f"Rollback {results['overall_status']}"
                )
        
        except Exception as e:
            self.logger.error(f"Rollback execution failed: {e}")
            results["overall_status"] = "failed"
            results["error"] = str(e)
            results["completed_at"] = datetime.now().isoformat()
            
            if not dry_run:
                await self._send_rollback_notification(
                    "failed", plan, f"Rollback failed: {str(e)}"
                )
        
        finally:
            if not dry_run and plan.plan_id in self.active_rollbacks:
                del self.active_rollbacks[plan.plan_id]
        
        return results
    
    async def _simulate_action(self, action: RollbackAction) -> Dict[str, Any]:
        """Simulate action execution for dry run"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "description": action.description,
            "status": "simulated",
            "duration": action.estimated_duration,
            "executed_at": datetime.now().isoformat()
        }
    
    async def _execute_action(
        self, 
        action: RollbackAction, 
        plan: RollbackPlan
    ) -> Dict[str, Any]:
        """Execute single rollback action"""
        start_time = datetime.now()
        action.status = RollbackStatus.IN_PROGRESS
        action.executed_at = start_time
        
        try:
            self.logger.info(f"Executing action: {action.action_id}")
            
            # Execute based on action type
            if action.action_type == "dns":
                success = await self._execute_dns_restore(action, plan)
            elif action.action_type == "redirect":
                success = await self._execute_redirect_restore(action, plan)
            elif action.action_type == "config":
                success = await self._execute_config_restore(action, plan)
            elif action.action_type == "database":
                success = await self._execute_database_restore(action, plan)
            elif action.action_type == "cache":
                success = await self._execute_cache_purge(action, plan)
            elif action.action_type == "seo":
                success = await self._execute_seo_notify(action, plan)
            else:
                success = False
                action.execution_log.append(f"Unknown action type: {action.action_type}")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                action.status = RollbackStatus.COMPLETED
                return {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "description": action.description,
                    "status": "completed",
                    "duration": duration,
                    "executed_at": start_time.isoformat()
                }
            else:
                action.status = RollbackStatus.FAILED
                return {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "description": action.description,
                    "status": "failed",
                    "duration": duration,
                    "executed_at": start_time.isoformat(),
                    "logs": action.execution_log
                }
        
        except Exception as e:
            action.status = RollbackStatus.FAILED
            action.execution_log.append(f"Execution error: {str(e)}")
            
            return {
                "action_id": action.action_id,
                "status": "failed",
                "error": str(e),
                "executed_at": start_time.isoformat(),
                "logs": action.execution_log
            }
    
    async def _execute_dns_restore(
        self, 
        action: RollbackAction, 
        plan: RollbackPlan
    ) -> bool:
        """Execute DNS restore action"""
        try:
            dns_records = plan.pre_migration_snapshot.dns_records
            action.execution_log.append(f"Restoring {len(dns_records)} DNS record types")
            
            # Mock DNS API calls
            await asyncio.sleep(1)  # Simulate API call
            
            action.execution_log.append("DNS records restored successfully")
            return True
        
        except Exception as e:
            action.execution_log.append(f"DNS restore failed: {str(e)}")
            return False
    
    async def _execute_redirect_restore(
        self, 
        action: RollbackAction, 
        plan: RollbackPlan
    ) -> bool:
        """Execute redirect restore action"""
        try:
            redirects = plan.pre_migration_snapshot.redirects
            action.execution_log.append(f"Restoring {len(redirects)} redirect rules")
            
            # Mock redirect configuration update
            await asyncio.sleep(0.5)
            
            action.execution_log.append("Redirect rules restored successfully")
            return True
        
        except Exception as e:
            action.execution_log.append(f"Redirect restore failed: {str(e)}")
            return False
    
    async def _execute_config_restore(
        self, 
        action: RollbackAction, 
        plan: RollbackPlan
    ) -> bool:
        """Execute configuration restore action"""
        try:
            backups = plan.pre_migration_snapshot.config_backups
            action.execution_log.append(f"Restoring {len(backups)} config files")
            
            for original_path, backup_path in backups.items():
                if Path(backup_path).exists():
                    # In practice, would copy backup back to original location
                    action.execution_log.append(f"Restored {original_path}")
            
            action.execution_log.append("Configuration files restored successfully")
            return True
        
        except Exception as e:
            action.execution_log.append(f"Config restore failed: {str(e)}")
            return False
    
    async def _execute_database_restore(
        self, 
        action: RollbackAction, 
        plan: RollbackPlan
    ) -> bool:
        """Execute database restore action"""
        try:
            backup_path = plan.pre_migration_snapshot.database_backup
            if not backup_path:
                action.execution_log.append("No database backup available")
                return False
            
            action.execution_log.append(f"Restoring database from {backup_path}")
            
            # Mock database restore
            await asyncio.sleep(2)  # Simulate restore time
            
            action.execution_log.append("Database restored successfully")
            return True
        
        except Exception as e:
            action.execution_log.append(f"Database restore failed: {str(e)}")
            return False
    
    async def _execute_cache_purge(
        self, 
        action: RollbackAction, 
        plan: RollbackPlan
    ) -> bool:
        """Execute cache purge action"""
        try:
            action.execution_log.append("Purging CDN and application caches")
            
            # Mock cache purge operations
            await asyncio.sleep(0.5)
            
            action.execution_log.append("All caches purged successfully")
            return True
        
        except Exception as e:
            action.execution_log.append(f"Cache purge failed: {str(e)}")
            return False
    
    async def _execute_seo_notify(
        self, 
        action: RollbackAction, 
        plan: RollbackPlan
    ) -> bool:
        """Execute SEO notification action"""
        try:
            action.execution_log.append("Notifying search engines of changes")
            
            # Mock search engine notifications
            sitemap_urls = plan.pre_migration_snapshot.sitemap_urls
            action.execution_log.append(f"Submitting {len(sitemap_urls)} URLs to search engines")
            
            await asyncio.sleep(0.3)
            
            action.execution_log.append("Search engines notified successfully")
            return True
        
        except Exception as e:
            action.execution_log.append(f"SEO notification failed: {str(e)}")
            return False
    
    async def _send_rollback_notification(
        self, 
        event_type: str, 
        plan: RollbackPlan, 
        message: str
    ):
        """Send rollback notifications"""
        notification = {
            "event": event_type,
            "plan_id": plan.plan_id,
            "migration_id": plan.migration_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "trigger": plan.trigger_condition.value
        }
        
        # Log notification
        self.logger.info(f"Notification: {notification}")
        
        # In practice, would send to configured channels
        if self.config["notifications"]["email_alerts"]:
            pass  # Send email
        
        for webhook in self.config["notifications"]["slack_webhooks"]:
            pass  # Send Slack notification
    
    def get_rollback_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of rollback execution"""
        if plan_id in self.active_rollbacks:
            plan = self.active_rollbacks[plan_id]
            return {
                "plan_id": plan_id,
                "status": "in_progress",
                "progress": {
                    "completed_actions": len([
                        a for a in plan.rollback_actions 
                        if a.status == RollbackStatus.COMPLETED
                    ]),
                    "total_actions": len(plan.rollback_actions),
                    "current_action": next(
                        (a.description for a in plan.rollback_actions 
                         if a.status == RollbackStatus.IN_PROGRESS),
                        None
                    )
                }
            }
        return None
    
    def generate_rollback_report(
        self, 
        execution_results: Dict[str, Any]
    ) -> str:
        """Generate detailed rollback execution report"""
        
        template_str = """
# Rollback Execution Report

**Plan ID:** {{ plan_id }}
**Execution Status:** {{ overall_status|upper }}
**Started:** {{ started_at }}
**Completed:** {{ completed_at }}
**Dry Run:** {{ dry_run }}

## Summary
- **Total Actions:** {{ actions|length }}
- **Successful:** {{ successful_count }}
- **Failed:** {{ failed_count }}

## Action Details
{% for action in actions %}
### {{ action.action_id }}
- **Type:** {{ action.action_type }}
- **Status:** {{ action.status }}
- **Duration:** {{ action.duration }}s
{% if action.logs %}
- **Logs:**
{% for log in action.logs %}
  - {{ log }}
{% endfor %}
{% endif %}
{% if action.error %}
- **Error:** {{ action.error }}
{% endif %}

{% endfor %}

## Recommendations
{% if failed_count > 0 %}
⚠️ **{{ failed_count }} actions failed.** Review error logs and consider manual intervention.
{% endif %}

{% if overall_status == 'completed' %}
✅ **Rollback completed successfully.** Monitor key metrics for the next 24 hours.
{% endif %}

---
*Report generated at {{ now }}*
        """
        
        template = Template(template_str.strip())
        
        successful_count = len([a for a in execution_results["actions"] if a["status"] == "completed"])
        failed_count = len([a for a in execution_results["actions"] if a["status"] == "failed"])
        
        return template.render(
            plan_id=execution_results["plan_id"],
            overall_status=execution_results["overall_status"],
            started_at=execution_results["started_at"],
            completed_at=execution_results.get("completed_at", "In Progress"),
            dry_run=execution_results["dry_run"],
            actions=execution_results["actions"],
            successful_count=successful_count,
            failed_count=failed_count,
            now=datetime.now().isoformat()
        )


async def main():
    """Demo usage of Rollback Automation"""
    
    async with RollbackAutomation() as rollback_system:
        print("SEO Migration Rollback Automation Demo")
        
        # Create migration snapshot
        snapshot = rollback_system.create_migration_snapshot(
            migration_id="migration_2024_01",
            description="E-commerce site redesign migration"
        )
        
        print(f"\n📸 Snapshot created: {snapshot.snapshot_id}")
        print(f"Captured {len(snapshot.redirects)} redirect rules")
        print(f"Captured {len(snapshot.sitemap_urls)} sitemap URLs")
        
        # Create rollback plan
        plan = rollback_system.create_rollback_plan(
            migration_id="migration_2024_01",
            snapshot=snapshot,
            trigger=RollbackTrigger.TRAFFIC_DROP,
            trigger_details="25% traffic drop detected in organic search"
        )
        
        print(f"\n📋 Rollback plan created: {plan.plan_id}")
        print(f"Actions: {len(plan.rollback_actions)}")
        print(f"Estimated duration: {plan.estimated_total_duration}s")
        
        # Execute dry run
        print(f"\n🧪 Executing dry run...")
        dry_run_results = await rollback_system.execute_rollback(plan, dry_run=True)
        
        print(f"Dry run status: {dry_run_results['overall_status']}")
        print(f"Actions simulated: {len(dry_run_results['actions'])}")
        
        # Generate report
        report = rollback_system.generate_rollback_report(dry_run_results)
        print(f"\n📄 Rollback Report Generated")
        
        # In production, would execute actual rollback:
        # results = await rollback_system.execute_rollback(plan, dry_run=False)


if __name__ == "__main__":
    asyncio.run(main())