"""
Service Factory - Dependency Injection and Service Management

This module implements a centralized service factory for dependency injection,
service lifecycle management, and configuration-driven service creation.
"""

import logging
import threading
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Type, Any, Optional, Callable, TypeVar, Generic
from threading import Lock, RLock
import time

from .event_bus import EventBusService, EventTypes
from .config_manager import ConfigurationManager, Configuration
from ..utils.exceptions import (
    InitializationError,
    AstirError
)


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceStatus(Enum):
    """Service status enumeration."""
    NOT_CREATED = "not_created"
    CREATING = "creating"
    READY = "ready"
    ERROR = "error"
    DISPOSED = "disposed"


@dataclass
class ServiceMetadata:
    """Service registration metadata."""
    service_type: Type
    interface_type: Optional[Type] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    dependencies: List[Type] = field(default_factory=list)
    factory_method: Optional[Callable] = None
    configuration_section: Optional[str] = None
    description: str = ""
    version: str = "1.0.0"


@dataclass
class ServiceInstance:
    """Service instance container."""
    instance: Any
    metadata: ServiceMetadata
    status: ServiceStatus
    created_at: float
    last_accessed: float
    access_count: int = 0
    error_message: Optional[str] = None


@dataclass
class ServiceStatistics:
    """Service factory statistics."""
    total_services_registered: int
    total_instances_created: int
    singleton_instances: int
    transient_instances: int
    failed_creations: int
    average_creation_time_ms: float
    total_memory_usage_mb: float


class ServiceFactoryError(AstirError):
    """Service factory specific errors."""
    pass


class DependencyResolutionError(ServiceFactoryError):
    """Dependency resolution errors."""
    pass


class CircularDependencyError(DependencyResolutionError):
    """Circular dependency detection error."""
    pass


T = TypeVar('T')


class IServiceFactory(ABC):
    """Service factory interface."""
    
    @abstractmethod
    def register(self, service_type: Type[T], metadata: ServiceMetadata) -> None:
        """Register a service type with the factory."""
        pass
    
    @abstractmethod
    def create(self, service_type: Type[T]) -> T:
        """Create or retrieve a service instance."""
        pass
    
    @abstractmethod
    def get(self, service_type: Type[T]) -> Optional[T]:
        """Get an existing service instance."""
        pass


class ServiceFactory(IServiceFactory):
    """
    Centralized service factory for dependency injection and lifecycle management.
    
    Provides service registration, automatic dependency resolution, lifecycle management,
    and configuration-driven service creation following the Factory pattern.
    """
    
    def __init__(self, event_bus: Optional[EventBusService] = None, 
                 config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the Service Factory.
        
        Args:
            event_bus: Event bus for publishing service lifecycle events
            config_manager: Configuration manager for service settings
        """
        self._event_bus = event_bus
        self._config_manager = config_manager
        self._logger = logging.getLogger(__name__)
        
        # Service registry and instances
        self._service_registry: Dict[Type, ServiceMetadata] = {}
        self._service_instances: Dict[Type, ServiceInstance] = {}
        self._interface_mappings: Dict[Type, Type] = {}
        
        # Thread safety
        self._registry_lock = RLock()
        self._instance_lock = RLock()
        
        # Dependency resolution
        self._resolution_stack: List[Type] = []
        self._resolution_lock = Lock()
        
        # Statistics
        self._stats = {
            'total_services_registered': 0,
            'total_instances_created': 0,
            'singleton_instances': 0,
            'transient_instances': 0,
            'failed_creations': 0,
            'creation_times': [],
            'total_memory_usage': 0
        }
        self._stats_lock = Lock()
        
        # Factory state
        self._is_initialized = False
        self._is_disposed = False
        
        # Service registry
        self._service_registry: Dict[Type, Type] = {}
    
    def initialize(self) -> bool:
        """
        Initialize the service factory.
        
        Returns:
            bool: True if initialization successful
            
        Raises:
            InitializationError: If initialization fails
        """
        if self._is_initialized:
            self._logger.warning("Service Factory already initialized")
            return True
        
        try:
            # Register core services
            self._register_core_services()
            
            # Validate service registry
            self._validate_service_registry()
            
            self._is_initialized = True
            self._logger.info("Service Factory initialized successfully")
            
            # Publish initialization event
            if self._event_bus:
                self._event_bus.publish(EventTypes.SYSTEM_READY, {
                    'service': 'ServiceFactory',
                    'registered_services': len(self._service_registry),
                    'timestamp': time.time()
                })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Service Factory: {e}")
            raise InitializationError(f"Service factory initialization failed: {e}")
    
    def register(self, service_type: Type[T], metadata: ServiceMetadata) -> None:
        """
        Register a service type with the factory.
        
        Args:
            service_type: The service class type
            metadata: Service registration metadata
            
        Raises:
            ServiceFactoryError: If registration fails
        """
        if self._is_disposed:
            raise ServiceFactoryError("Service factory has been disposed")
        
        with self._registry_lock:
            if service_type in self._service_registry:
                self._logger.warning(f"Service {service_type.__name__} already registered, updating")
            
            # Validate service type
            self._validate_service_type(service_type, metadata)
            
            # Register service
            self._service_registry[service_type] = metadata
            
            # Register interface mapping if provided
            if metadata.interface_type:
                self._interface_mappings[metadata.interface_type] = service_type
            
            with self._stats_lock:
                self._stats['total_services_registered'] += 1
            
            self._logger.debug(f"Registered service: {service_type.__name__}")
            
            # Publish registration event
            if self._event_bus:
                self._event_bus.publish("SERVICE_REGISTERED", {
                    'service_type': service_type.__name__,
                    'lifetime': metadata.lifetime.value,
                    'dependencies': [dep.__name__ for dep in metadata.dependencies],
                    'timestamp': time.time()
                })
    
    def create(self, service_type: Type[T]) -> T:
        """
        Create or retrieve a service instance.
        
        Args:
            service_type: The service type to create
            
        Returns:
            Service instance
            
        Raises:
            DependencyResolutionError: If dependencies cannot be resolved
            ServiceFactoryError: If service creation fails
        """
        if self._is_disposed:
            raise ServiceFactoryError("Service factory has been disposed")
        
        if not self._is_initialized:
            raise ServiceFactoryError("Service factory not initialized")
        

        
        # Resolve interface to concrete type
        concrete_type = self._resolve_concrete_type(service_type)
        
        # Get service metadata
        metadata = self._get_service_metadata(concrete_type)
        
        # Handle singleton services
        if metadata.lifetime == ServiceLifetime.SINGLETON:
            with self._instance_lock:
                if concrete_type in self._service_instances:
                    instance_info = self._service_instances[concrete_type]
                    if instance_info.status == ServiceStatus.READY:
                        instance_info.last_accessed = time.time()
                        instance_info.access_count += 1
                        return instance_info.instance
                    elif instance_info.status == ServiceStatus.ERROR:
                        raise ServiceFactoryError(f"Service {concrete_type.__name__} is in error state: {instance_info.error_message}")
        
        # Create new instance
        return self._create_instance(concrete_type, metadata)
    
    def get(self, service_type: Type[T]) -> Optional[T]:
        """
        Get an existing service instance.
        
        Args:
            service_type: The service type to retrieve
            
        Returns:
            Service instance or None if not found
        """
        if self._is_disposed:
            return None
        

        
        # Resolve interface to concrete type
        concrete_type = self._resolve_concrete_type(service_type)
        
        with self._instance_lock:
            if concrete_type in self._service_instances:
                instance_info = self._service_instances[concrete_type]
                if instance_info.status == ServiceStatus.READY:
                    instance_info.last_accessed = time.time()
                    instance_info.access_count += 1
                    return instance_info.instance
        
        return None
    
    def dispose_service(self, service_type: Type) -> bool:
        """
        Dispose a service instance.
        
        Args:
            service_type: The service type to dispose
            
        Returns:
            bool: True if disposed successfully
        """
        concrete_type = self._resolve_concrete_type(service_type)
        
        with self._instance_lock:
            if concrete_type in self._service_instances:
                instance_info = self._service_instances[concrete_type]
                
                # Call dispose method if available
                if hasattr(instance_info.instance, 'dispose'):
                    try:
                        instance_info.instance.dispose()
                    except Exception as e:
                        self._logger.warning(f"Error disposing service {concrete_type.__name__}: {e}")
                
                # Update status
                instance_info.status = ServiceStatus.DISPOSED
                
                # Remove from instances for singletons
                metadata = self._get_service_metadata(concrete_type)
                if metadata.lifetime == ServiceLifetime.SINGLETON:
                    del self._service_instances[concrete_type]
                
                # Publish disposal event
                if self._event_bus:
                    self._event_bus.publish("SERVICE_DISPOSED", {
                        'service_type': concrete_type.__name__,
                        'timestamp': time.time()
                    })
                
                return True
        
        return False
    

    
    def get_registered_services(self) -> List[Type]:
        """Get list of all registered service types."""
        with self._registry_lock:
            return list(self._service_registry.keys())
    
    def get_service_metadata(self, service_type: Type) -> Optional[ServiceMetadata]:
        """Get metadata for a service type."""
        concrete_type = self._resolve_concrete_type(service_type)
        return self._service_registry.get(concrete_type)
    
    def get_statistics(self) -> ServiceStatistics:
        """Get service factory statistics."""
        with self._stats_lock:
            avg_creation_time = (
                sum(self._stats['creation_times']) / len(self._stats['creation_times'])
                if self._stats['creation_times'] else 0.0
            )
            
            return ServiceStatistics(
                total_services_registered=self._stats['total_services_registered'],
                total_instances_created=self._stats['total_instances_created'],
                singleton_instances=self._stats['singleton_instances'],
                transient_instances=self._stats['transient_instances'],
                failed_creations=self._stats['failed_creations'],
                average_creation_time_ms=avg_creation_time,
                total_memory_usage_mb=self._stats['total_memory_usage']
            )
    
    def shutdown(self) -> None:
        """Shutdown the service factory and dispose all services."""
        if self._is_disposed:
            return
        
        self._logger.info("Shutting down Service Factory...")
        
        # Dispose all singleton services
        with self._instance_lock:
            for service_type in list(self._service_instances.keys()):
                self.dispose_service(service_type)
        
        # Clear registrations
        with self._registry_lock:
            self._service_registry.clear()
            self._interface_mappings.clear()
        
        self._is_disposed = True
        self._logger.info("Service Factory shutdown complete")
    
    def _register_core_services(self) -> None:
        """Register core system services."""
        # Note: Core services are registered by the main application
        # This method is for any built-in service registrations
        pass
    
    def _validate_service_registry(self) -> None:
        """Validate the service registry for consistency."""
        with self._registry_lock:
            for service_type, metadata in self._service_registry.items():
                # Check for circular dependencies
                self._check_circular_dependencies(service_type, set())
                
                # Validate dependencies exist
                for dep in metadata.dependencies:
                    if dep not in self._service_registry and dep not in self._interface_mappings:
                        self._logger.warning(f"Service {service_type.__name__} depends on unregistered service {dep.__name__}")
    
    def _validate_service_type(self, service_type: Type, metadata: ServiceMetadata) -> None:
        """Validate a service type and its metadata."""
        if not inspect.isclass(service_type):
            raise ServiceFactoryError(f"Service type must be a class: {service_type}")
        
        # Validate factory method if provided
        if metadata.factory_method and not callable(metadata.factory_method):
            raise ServiceFactoryError(f"Factory method must be callable for {service_type.__name__}")
    
    def _resolve_concrete_type(self, service_type: Type) -> Type:
        """Resolve interface type to concrete implementation."""
        return self._interface_mappings.get(service_type, service_type)
    
    def _get_service_metadata(self, service_type: Type) -> ServiceMetadata:
        """Get service metadata, raising error if not found."""
        with self._registry_lock:
            if service_type not in self._service_registry:
                raise ServiceFactoryError(f"Service {service_type.__name__} is not registered")
            return self._service_registry[service_type]
    
    def _create_instance(self, service_type: Type, metadata: ServiceMetadata) -> Any:
        """Create a new service instance with dependency injection."""
        start_time = time.time()
        
        try:
            with self._resolution_lock:
                # Check for circular dependencies
                if service_type in self._resolution_stack:
                    cycle = " -> ".join([t.__name__ for t in self._resolution_stack] + [service_type.__name__])
                    raise CircularDependencyError(f"Circular dependency detected: {cycle}")
                
                self._resolution_stack.append(service_type)
                
                try:
                    # Mark as creating
                    instance_info = ServiceInstance(
                        instance=None,
                        metadata=metadata,
                        status=ServiceStatus.CREATING,
                        created_at=start_time,
                        last_accessed=start_time
                    )
                    
                    if metadata.lifetime == ServiceLifetime.SINGLETON:
                        with self._instance_lock:
                            self._service_instances[service_type] = instance_info
                    
                    # Resolve dependencies
                    dependencies = self._resolve_dependencies(metadata.dependencies)
                    
                    # Create instance
                    if metadata.factory_method:
                        instance = metadata.factory_method(**dependencies)
                    else:
                        instance = self._create_with_constructor(service_type, dependencies)
                    
                    # Initialize if method exists
                    if hasattr(instance, 'initialize'):
                        instance.initialize()
                    
                    # Update instance info
                    instance_info.instance = instance
                    instance_info.status = ServiceStatus.READY
                    
                    # Update statistics
                    creation_time = (time.time() - start_time) * 1000  # Convert to ms
                    with self._stats_lock:
                        self._stats['total_instances_created'] += 1
                        self._stats['creation_times'].append(creation_time)
                        if metadata.lifetime == ServiceLifetime.SINGLETON:
                            self._stats['singleton_instances'] += 1
                        else:
                            self._stats['transient_instances'] += 1
                    
                    # Publish creation event
                    if self._event_bus:
                        self._event_bus.publish("SERVICE_CREATED", {
                            'service_type': service_type.__name__,
                            'lifetime': metadata.lifetime.value,
                            'creation_time_ms': creation_time,
                            'timestamp': time.time()
                        })
                    
                    self._logger.debug(f"Created service {service_type.__name__} in {creation_time:.2f}ms")
                    return instance
                    
                finally:
                    self._resolution_stack.remove(service_type)
        
        except Exception as e:
            # Update statistics
            with self._stats_lock:
                self._stats['failed_creations'] += 1
            
            # Update instance status if singleton
            if metadata.lifetime == ServiceLifetime.SINGLETON:
                with self._instance_lock:
                    if service_type in self._service_instances:
                        self._service_instances[service_type].status = ServiceStatus.ERROR
                        self._service_instances[service_type].error_message = str(e)
            
            # Publish error event
            if self._event_bus:
                self._event_bus.publish("SERVICE_ERROR", {
                    'service_type': service_type.__name__,
                    'error': str(e),
                    'timestamp': time.time()
                })
            
            self._logger.error(f"Failed to create service {service_type.__name__}: {e}")
            raise ServiceFactoryError(f"Failed to create service {service_type.__name__}: {e}")
    
    def _resolve_dependencies(self, dependencies: List[Type]) -> Dict[str, Any]:
        """Resolve service dependencies."""
        resolved = {}
        
        for dep_type in dependencies:
            # Special handling for configuration
            if dep_type == Configuration or (hasattr(dep_type, '__name__') and 'Config' in dep_type.__name__):
                if self._config_manager:
                    config = self._config_manager.get_config()
                    if hasattr(config, dep_type.__name__.lower().replace('config', '')):
                        resolved[dep_type.__name__.lower()] = getattr(config, dep_type.__name__.lower().replace('config', ''))
                    else:
                        resolved[dep_type.__name__.lower()] = config
                continue
            
            # Resolve service dependency
            dep_instance = self.create(dep_type)
            param_name = dep_type.__name__.lower().replace('service', '')
            resolved[param_name] = dep_instance
        
        return resolved
    
    def _create_with_constructor(self, service_type: Type, dependencies: Dict[str, Any]) -> Any:
        """Create service instance using constructor injection."""
        # Get constructor signature
        sig = inspect.signature(service_type.__init__)
        
        # Match dependencies to constructor parameters
        constructor_args = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Try to find matching dependency
            if param_name in dependencies:
                constructor_args[param_name] = dependencies[param_name]
            elif param.default != inspect.Parameter.empty:
                # Use default value
                continue
            else:
                # Try to match by type annotation
                if param.annotation != inspect.Parameter.empty:
                    for dep_name, dep_instance in dependencies.items():
                        if isinstance(dep_instance, param.annotation):
                            constructor_args[param_name] = dep_instance
                            break
        
        return service_type(**constructor_args)
    
    def _check_circular_dependencies(self, service_type: Type, visited: set) -> None:
        """Check for circular dependencies in service graph."""
        if service_type in visited:
            raise CircularDependencyError(f"Circular dependency detected involving {service_type.__name__}")
        
        visited.add(service_type)
        
        try:
            metadata = self._service_registry.get(service_type)
            if metadata:
                for dep in metadata.dependencies:
                    concrete_dep = self._resolve_concrete_type(dep)
                    if concrete_dep in self._service_registry:
                        self._check_circular_dependencies(concrete_dep, visited)
        finally:
            visited.remove(service_type)


# Global service factory instance
service_factory: Optional[ServiceFactory] = None


def get_service_factory() -> ServiceFactory:
    """Get the global service factory instance."""
    global service_factory
    if service_factory is None:
        raise ServiceFactoryError("Service factory not initialized")
    return service_factory


def initialize_service_factory(event_bus: Optional[EventBusService] = None,
                             config_manager: Optional[ConfigurationManager] = None) -> ServiceFactory:
    """Initialize the global service factory."""
    global service_factory
    if service_factory is None:
        service_factory = ServiceFactory(event_bus, config_manager)
        service_factory.initialize()
    return service_factory
