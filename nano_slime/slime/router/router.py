"""
Phase 5: Slime Router

이 파일은 SGLang 엔진들을 위한 HTTP 로드 밸런서를 구현합니다.
원본 slime/router/router.py에서 핵심만 추출.

학습 포인트:
1. 최소 활성 요청 기반 로드 밸런싱
2. 헬스 체크와 장애 워커 격리
3. FastAPI 프록시 패턴
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SlimeRouter:
    """
    SGLang 엔진 로드 밸런서

    역할:
    1. 요청을 여러 SGLang 엔진에 분배
    2. 최소 활성 요청 기반 라우팅
    3. 장애 워커 감지 및 격리
    4. 헬스 체크 수행

    라우팅 전략:
    - Least Connections: 가장 적은 활성 요청을 가진 워커 선택
    - 장애 워커는 dead_workers 집합에 격리
    """

    def __init__(self, args, verbose: bool = False):
        """
        Args:
            args: 설정
            verbose: 상세 로깅 여부
        """
        self.args = args
        self.verbose = verbose

        # URL → 활성 요청 수
        self.worker_request_counts: dict[str, int] = {}

        # URL → 연속 실패 횟수
        self.worker_failure_counts: dict[str, int] = {}

        # 격리된 (죽은) 워커들
        self.dead_workers: set[str] = set()

        # HTTP 클라이언트 (실제 구현에서는 httpx 사용)
        self.client = None

    def add_worker(self, worker_url: str) -> dict[str, Any]:
        """
        워커 추가

        Args:
            worker_url: SGLang 엔진 URL

        Returns:
            상태 정보
        """
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            if self.verbose:
                logger.info(f"[slime-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": list(self.worker_request_counts.keys())}

    def list_workers(self) -> dict[str, list[str]]:
        """등록된 워커 목록 반환"""
        return {"urls": list(self.worker_request_counts.keys())}

    def _use_url(self) -> str:
        """
        최소 활성 요청 워커 선택

        Returns:
            선택된 워커 URL

        알고리즘:
        1. 죽은 워커 제외
        2. 남은 워커 중 최소 요청 수 선택
        3. 선택된 워커의 요청 수 증가
        """
        if not self.dead_workers:
            # 모든 워커가 정상
            url = min(self.worker_request_counts, key=self.worker_request_counts.get)
        else:
            # 죽은 워커 제외
            valid_workers = {
                url: count
                for url, count in self.worker_request_counts.items()
                if url not in self.dead_workers
            }
            if not valid_workers:
                raise RuntimeError("No healthy workers available in the pool")
            url = min(valid_workers, key=valid_workers.get)

        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url: str) -> None:
        """
        요청 완료 처리

        Args:
            url: 완료된 워커 URL
        """
        assert url in self.worker_request_counts, f"URL {url} not recognized"
        self.worker_request_counts[url] -= 1
        assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"

    async def proxy(self, request: Any, path: str) -> Any:
        """
        요청 프록시

        Args:
            request: HTTP 요청
            path: 요청 경로

        Returns:
            프록시된 응답

        핵심 로직:
        1. _use_url()로 워커 선택
        2. 워커에 요청 전달
        3. _finish_url()로 완료 처리
        """
        worker_url = self._use_url()

        try:
            # 실제 구현에서는 httpx로 프록시
            # response = await self.client.request(...)
            # return response
            pass
        finally:
            self._finish_url(worker_url)

    async def _check_worker_health(self, url: str) -> tuple[str, bool]:
        """
        워커 헬스 체크

        Args:
            url: 워커 URL

        Returns:
            (url, is_healthy)
        """
        try:
            # 실제 구현에서는 /health 엔드포인트 호출
            # response = await self.client.get(f"{url}/health", timeout=5.0)
            # return url, response.status_code == 200
            return url, True
        except Exception as e:
            logger.debug(f"[slime-router] Worker {url} health check failed: {e}")
            return url, False

    async def _health_check_loop(self) -> None:
        """
        백그라운드 헬스 체크 루프

        동작:
        1. 주기적으로 모든 워커 헬스 체크
        2. 연속 실패 시 dead_workers에 추가
        3. 복구된 워커는 다시 활성화 (미구현)
        """
        interval = getattr(self.args, "rollout_health_check_interval", 10)
        threshold = getattr(self.args, "slime_router_health_check_failure_threshold", 3)

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(
                    *(self._check_worker_health(url) for url in urls)
                )

                for url, is_healthy in results:
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[slime-router] Worker {url} failed {threshold} "
                                f"consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                    else:
                        self.worker_failure_counts[url] = 0

            except asyncio.CancelledError:
                logger.warning("[slime-router] Health check loop cancelled.")
                raise
            except Exception as e:
                logger.error(f"[slime-router] Error in health check loop: {e}")
                await asyncio.sleep(5)


def run_router(args) -> None:
    """
    Router 실행

    Args:
        args: 설정 (host, port 등)
    """
    try:
        import uvicorn
        from fastapi import FastAPI

        router = SlimeRouter(args, verbose=True)
        app = FastAPI()

        # 라우트 등록
        @app.post("/add_worker")
        async def add_worker(url: str):
            return router.add_worker(url)

        @app.get("/list_workers")
        async def list_workers():
            return router.list_workers()

        @app.api_route("/{path:path}", methods=["GET", "POST"])
        async def proxy(request, path: str):
            return await router.proxy(request, path)

        uvicorn.run(
            app,
            host=getattr(args, "sglang_router_ip", "0.0.0.0"),
            port=getattr(args, "sglang_router_port", 30000),
            log_level="info",
        )
    except ImportError:
        logger.warning("FastAPI/uvicorn not installed. Router not available.")
