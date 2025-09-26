#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLLM Load Monitoring Tool
Real-time monitoring of multiple VLLM service load status
"""

import asyncio
import aiohttp
import time
import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.columns import Columns

# Load .env file
load_dotenv()

class ServiceStatus(Enum):
    HEALTHY = "Healthy"
    UNHEALTHY = "Unhealthy"
    UNKNOWN = "Unknown"

@dataclass
class VLLMService:
    url: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    response_time: float = 0.0
    error_message: str = ""
    last_check: datetime = None
    model_info: Dict[str, Any] = None
    running_requests: int = 0
    waiting_requests: int = 0
    swapped_requests: int = 0
    metrics_available: bool = False
    weight: float = 0.0

class VLLMMonitor:
    def __init__(self, base_urls: List[str], model_name: str = "8001vllm"):
        self.services = [VLLMService(url=url.strip()) for url in base_urls if url.strip()]
        self.model_name = model_name
        self.session = None
        self.url_weights = {}  # Store weight information
        self.console = Console()
        
        # Initialize weights
        for url in base_urls:
            self.url_weights[url.strip()] = 1.0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_service_health(self, service: VLLMService) -> None:
        """Check health status of a single service"""
        start_time = time.time()
        try:
            # Check model list endpoint
            models_url = f"{service.url}/models"
            async with self.session.get(models_url) as response:
                service.response_time = time.time() - start_time
                service.last_check = datetime.now()
                
                if response.status == 200:
                    data = await response.json()
                    service.model_info = data
                    service.status = ServiceStatus.HEALTHY
                    service.error_message = ""
                    
                    # Get metrics information
                    await self.get_service_metrics(service)
                else:
                    service.status = ServiceStatus.UNHEALTHY
                    service.error_message = f"HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            service.response_time = time.time() - start_time
            service.last_check = datetime.now()
            service.status = ServiceStatus.UNHEALTHY
            service.error_message = "Request timeout"
        except Exception as e:
            service.response_time = time.time() - start_time
            service.last_check = datetime.now()
            service.status = ServiceStatus.UNHEALTHY
            service.error_message = str(e)
    
    async def get_service_metrics(self, service: VLLMService) -> None:
        """Get service metrics information"""
        try:
            # Build metrics endpoint URL - fix URL construction logic
            if service.url.endswith('/v1'):
                base_url = service.url[:-3]  # Remove trailing '/v1'
            else:
                base_url = service.url.rstrip('/')
            metrics_url = f"{base_url}/metrics"
            
            async with self.session.get(metrics_url) as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    service.metrics_available = True
                    
                    # Parse Prometheus format metrics
                    service.running_requests = self.parse_metric_value(metrics_text, 'vllm:num_requests_running')
                    service.waiting_requests = self.parse_metric_value(metrics_text, 'vllm:num_requests_waiting')
                    service.swapped_requests = self.parse_metric_value(metrics_text, 'vllm:num_requests_swapped')
                else:
                    service.metrics_available = False
                    # Record specific HTTP status code for debugging
                    if '8001' in service.url:
                        service.error_message = f"Metrics endpoint returned HTTP {response.status}"
        except Exception as e:
            service.metrics_available = False
            service.running_requests = 0
            service.waiting_requests = 0
            service.swapped_requests = 0
            # Record specific exception information for debugging port 8001 issues
            if '8001' in service.url:
                service.error_message = f"Metrics retrieval failed: {str(e)}"
    
    def parse_metric_value(self, metrics_text: str, metric_name: str) -> int:
        """Parse specified metric value from Prometheus format metrics text"""
        try:
            # Find metric lines, support labeled format like: vllm:num_requests_running{engine="0",model_name="8001vllm"} 5.0
            # Or simple format: vllm:num_requests_running 5.0
            pattern = rf'^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.]+)'
            matches = re.findall(pattern, metrics_text, re.MULTILINE)
            if matches:
                # If there are multiple matches (multiple label combinations), return the first value
                # Or can sum them, here we choose the first one
                return int(float(matches[0]))
        except Exception:
            pass
        return 0
    
    async def check_all_services(self) -> None:
        """Check all services concurrently"""
        tasks = [self.check_service_health(service) for service in self.services]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update weights
        self.update_weights()
    
    def update_weights(self) -> None:
        """Update weight information"""
        # Get load of all available services
        loads = []
        available_services = []
        
        for service in self.services:
            if service.metrics_available and service.status == ServiceStatus.HEALTHY:
                total_load = service.running_requests + service.waiting_requests
                loads.append(total_load)
                available_services.append(service)
        
        if not available_services:
            return
        
        # Use the same weight calculation algorithm as healthbench_reward_fn.py
        if loads:
            # Calculate average load
            avg_load = sum(loads) / len(loads)
            # Calculate double average
            double_avg = avg_load * 2
            
            # Calculate weight for each service: double average minus specific load value, negative values treated as 0
            raw_weights = []
            for service in available_services:
                current_load = service.running_requests + service.waiting_requests
                weight = max(0, double_avg - current_load)
                raw_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(raw_weights)
            if total_weight > 0:
                for i, service in enumerate(available_services):
                    normalized_weight = raw_weights[i] / total_weight
                    service.weight = normalized_weight
                    self.url_weights[service.url] = normalized_weight
            else:
                # If all weights are 0, distribute evenly
                for service in available_services:
                    service.weight = 1.0 / len(available_services)
                    self.url_weights[service.url] = 1.0 / len(available_services)
        else:
            # If no load data, distribute weights evenly
            for service in available_services:
                service.weight = 1.0 / len(available_services)
                self.url_weights[service.url] = 1.0 / len(available_services)
        
        # Set weight to 0 for unavailable services
        for service in self.services:
            if not (service.metrics_available and service.status == ServiceStatus.HEALTHY):
                service.weight = 0.0
                self.url_weights[service.url] = 0.0
    
    def create_status_display(self) -> Panel:
        """Create status display panel"""
        # Statistics
        healthy_count = sum(1 for s in self.services if s.status == ServiceStatus.HEALTHY)
        total_count = len(self.services)
        total_running = sum(s.running_requests for s in self.services if s.metrics_available)
        total_waiting = sum(s.waiting_requests for s in self.services if s.metrics_available)
        total_swapped = sum(s.swapped_requests for s in self.services if s.metrics_available)
        
        # Create 4-column grid layout
        service_panels = []
        for service in self.services:
            # Handle URL display
            url_display = service.url.replace('http://', '').replace('/v1', '')
            
            # Status display
            if service.status == ServiceStatus.HEALTHY:
                status_text = "[green]✓ Healthy[/green]"
                border_style = "green"
            elif service.status == ServiceStatus.UNHEALTHY:
                status_text = "[red]✗ Unhealthy[/red]"
                border_style = "red"
            else:
                status_text = "[yellow]? Unknown[/yellow]"
                border_style = "yellow"
            
            # Load information
            if service.metrics_available:
                total_load = service.running_requests + service.waiting_requests
                load_text = str(total_load)
                running_text = str(service.running_requests)
                waiting_text = str(service.waiting_requests)
                weight_text = f"[cyan]{100 * service.weight:.1f}%[/cyan]"
            else:
                load_text = "N/A"
                running_text = "N/A"
                waiting_text = "N/A"
                weight_text = "0%"
            
            # Create load bar
            if service.metrics_available:
                total_load = service.running_requests + service.waiting_requests
                # Calculate load bar height (max 6 blocks)
                load_ratio = min(total_load / 512, 1.0)
                bar_height = int(load_ratio * 6)
                
                # Determine color
                if total_load > 512:
                    bar_color = "red"
                elif total_load < 256:
                    bar_color = "green"
                else:
                    bar_color = "yellow"
                
                # Create vertical load bar (bottom to top) - use thicker characters
                load_bar_lines = []
                for i in range(6):
                    if i < (6 - bar_height):
                        load_bar_lines.append("┃")
                    else:
                        load_bar_lines.append(f"[{bar_color}]██[/{bar_color}]")
                
                load_bar = "\n".join(load_bar_lines)
            else:
                load_bar = "\n".join(["┃"] * 6)
            
            # Create single service panel content (left info + right load bar)
            # Use fixed width to ensure alignment
            service_info = f"""[cyan]{url_display:<20}[/cyan]
{status_text}
[bold]Load:[/bold] {load_text}
[bold]Running:[/bold] {running_text}
[bold]Waiting:[/bold] {waiting_text}
[bold]Weight:[/bold] {weight_text}"""
            
            # Use Columns to display info and load bar side by side, reduce right whitespace
            from rich.columns import Columns as InnerColumns
            service_layout = InnerColumns(
                [service_info, load_bar],
                equal=False,
                expand=False,
                padding=(0, 1)
            )
            
            service_panel = Panel(
                service_layout,
                border_style=border_style,
                width=32,
                height=8
            )
            service_panels.append(service_panel)
        
        # Arrange service panels by rows, 4 per row
        rows = []
        for i in range(0, len(service_panels), 4):
            row_panels = service_panels[i:i+4]
            row = Columns(row_panels, equal=True, expand=True)
            rows.append(row)
        
        # If only one row, use directly; otherwise create multi-row layout
        if len(rows) == 1:
            columns = rows[0]
        else:
            # Create vertical layout containing multiple rows
            from rich.layout import Layout as RowLayout
            columns_layout = RowLayout()
            columns_layout.split_column(*[RowLayout(row) for row in rows])
            columns = columns_layout
        
        # Create statistics (remove swapped)
        total_load = total_running + total_waiting
        stats_line1 = f"[bold]Total Services:[/bold] {total_count} | [bold green]Healthy:[/bold green] {healthy_count} | [bold red]Unhealthy:[/bold red] {total_count - healthy_count}"
        stats_line2 = f"[bold]Total Requests:[/bold] [magenta]Load: {total_load}[/magenta] | [cyan]Running: {total_running}[/cyan] | [yellow]Waiting: {total_waiting}[/yellow]"
        
        # Create statistics panel
        stats_panel = Panel(
            f"{stats_line1}\n{stats_line2}",
            border_style="dim",
            height=4
        )
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(stats_panel, size=4),
            Layout(columns)
        )
        
        # Create panel
        title = f"VLLM Service Load Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: {self.model_name}"
        panel = Panel(
            layout,
            title=title,
            subtitle="Press Ctrl+C to exit monitoring",
            border_style="blue"
        )
        
        return panel
    
    async def start_monitoring(self, interval: int = 1) -> None:
        """Start monitoring"""
        print("Starting VLLM service monitoring...")
        
        try:
            with Live(self.create_status_display(), refresh_per_second=1, screen=True) as live:
                while True:
                    await self.check_all_services()
                    live.update(self.create_status_display())
                    await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
        except Exception as e:
            print(f"\nMonitoring error: {e}")

def parse_urls(url_string: str) -> List[str]:
    """Parse URL string"""
    urls = []
    for url in url_string.split(','):
        url = url.strip().strip('"').strip("'")
        if url:
            urls.append(url)
    return urls

async def main():
    # Get VLLM service URL list from environment variables
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    VLLM_MODEL = os.getenv("VLLM_MODEL", "8001vllm")
    
    # Parse URLs
    urls = parse_urls(VLLM_BASE_URL)
    
    if not urls:
        print("Error: No valid VLLM service URLs found")
        sys.exit(1)
    
    print(f"Found {len(urls)} VLLM services")
    
    # Start monitoring
    async with VLLMMonitor(urls, VLLM_MODEL) as monitor:
        await monitor.start_monitoring(interval=1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram exited")
    except Exception as e:
        print(f"Program error: {e}")
        sys.exit(1)