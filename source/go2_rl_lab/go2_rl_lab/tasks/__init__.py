"""Discover and register the Go2 velocity locomotion tasks."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, blacklist_pkgs=[".mdp"])
