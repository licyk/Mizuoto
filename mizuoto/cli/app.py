"""Entry point: ``get_app()`` registers every command; ``main()`` runs it with uniform error handling."""

import sys
import traceback

import typer
from typer import Abort, Exit

from mizuoto.cli.commands.asr import asr
from mizuoto.cli.commands.assets import asset_delete, asset_download, asset_list
from mizuoto.cli.commands.config import config_get, config_path, config_set, config_show
from mizuoto.cli.commands.datasets import (
    dataset_add,
    dataset_check,
    dataset_create,
    dataset_delete,
    dataset_edit_row,
    dataset_import,
    dataset_info,
    dataset_list,
    dataset_prepare,
    dataset_rows,
)
from mizuoto.cli.commands.doctor import doctor
from mizuoto.cli.commands.engines import engine_accept_licence, engine_disable, engine_enable, engine_info, engine_licence, engine_list, engine_op
from mizuoto.cli.commands.jobs import job_cancel, job_list, job_log, job_retry, job_show
from mizuoto.cli.commands.models import model_delete, model_download, model_import, model_info, model_list, model_speakers, model_spec, model_verify
from mizuoto.cli.commands.outputs import output_check, output_delete, output_info, output_list
from mizuoto.cli.commands.presets import preset_default, preset_delete, preset_list, preset_save
from mizuoto.cli.commands.projects import (
    project_cast,
    project_delete,
    project_export,
    project_import,
    project_list,
    project_new,
    project_render,
    project_script,
    project_set,
    project_show,
)
from mizuoto.cli.commands.remote import agent, remote_add, remote_list, remote_remove, remote_use
from mizuoto.cli.commands.runtime import runtime
from mizuoto.cli.commands.speak import speak
from mizuoto.cli.commands.system import env_vars, version
from mizuoto.cli.commands.train import train_delete, train_edit, train_export, train_list, train_methods, train_metrics, train_new, train_run, train_status
from mizuoto.cli.commands.transform import convert, edit
from mizuoto.cli.commands.voices import (
    design,
    voice_add,
    voice_add_ref,
    voice_add_weights,
    voice_design,
    voice_edit,
    voice_export,
    voice_fit,
    voice_from_output,
    voice_from_preset,
    voice_import,
    voice_info,
    voice_list,
    voice_prepare,
    voice_remove,
    voice_remove_ref,
    voice_remove_weights,
    voice_retranscribe,
    voice_set_transcript,
)
from mizuoto.cli.commands.webui import webui
from mizuoto.cli.factory import ClickException, report_error, typer_factory
from mizuoto.logger import setup_logging

logger = setup_logging()


def get_app() -> typer.Typer:
    """Build the Mizuoto command line. Every command is registered here, and nowhere else."""
    app = typer_factory("Open TTS engines in one place: speak, clone and design voices, and train")

    app.command(help="Start the server and open the web UI", name="webui")(webui)
    app.command(help="Show the version of Mizuoto and its main components", name="version")(version)
    app.command(help="Speak text with a model and a voice", name="speak")(speak)
    app.command(help="Transcribe a clip", name="asr")(asr)
    app.command(help="Speak with a voice made from a description", name="design")(design)
    app.command(help="Say what a recording says in another voice", name="convert")(convert)
    app.command(help="Replace parts of a recording with new speech", name="edit")(edit)
    app.command(help="Check torch, the engines and their extras on this machine", name="doctor")(doctor)
    app.command(help="List the loaded models, and unload them", name="runtime")(runtime)

    config_cli = typer_factory(help="Show and change settings")
    config_cli.command(help="Show the effective settings", name="show")(config_show)
    config_cli.command(help="Print one setting", name="get")(config_get)
    config_cli.command(help="Change one setting and save it", name="set")(config_set)
    config_cli.command(help="Print the path of the settings file", name="path")(config_path)
    app.add_typer(config_cli, name="config")

    engine_cli = typer_factory(help="The TTS engines")
    engine_cli.command(help="List the engines", name="list")(engine_list)
    engine_cli.command(help="Show an engine's variants, parameters and models", name="info")(engine_info)
    engine_cli.command(help="Print an engine's licence", name="licence")(engine_licence)
    engine_cli.command(help="Run an engine-specific operation", name="op")(engine_op)
    engine_cli.command(help="Accept a licence-gated engine's licence", name="accept-licence")(engine_accept_licence)
    engine_cli.command(help="Enable an engine", name="enable")(engine_enable)
    engine_cli.command(help="Disable an engine", name="disable")(engine_disable)
    app.add_typer(engine_cli, name="engine")

    env_cli = typer_factory(help="The environment variables Mizuoto reads")
    env_cli.command(help="List the environment variables Mizuoto reads", name="vars")(env_vars)
    app.add_typer(env_cli, name="env")

    model_cli = typer_factory(help="Engine models")
    model_cli.command(help="List models", name="list")(model_list)
    model_cli.command(help="Show a model", name="info")(model_info)
    model_cli.command(help="Show what a model takes: voices, languages, parameters", name="spec")(model_spec)
    model_cli.command(help="Download catalogue models", name="download")(model_download)
    model_cli.command(help="Check an installed model's files", name="verify")(model_verify)
    model_cli.command(help="Delete installed models", name="delete")(model_delete)
    model_cli.command(help="Add a model folder from elsewhere", name="import")(model_import)
    model_cli.command(help="List a model's preset speakers", name="speakers")(model_speakers)
    app.add_typer(model_cli, name="model")

    asset_cli = typer_factory(help="Files several models of an engine share")
    asset_cli.command(help="List the shared assets", name="list")(asset_list)
    asset_cli.command(help="Download shared assets", name="download")(asset_download)
    asset_cli.command(help="Delete a shared asset", name="delete")(asset_delete)
    app.add_typer(asset_cli, name="asset")

    voice_cli = typer_factory(help="The voice library")
    voice_cli.command(help="List voices", name="list")(voice_list)
    voice_cli.command(help="Show a voice", name="info")(voice_info)
    voice_cli.command(help="Add a voice from reference clips", name="add")(voice_add)
    voice_cli.command(help="Design a voice from a description", name="design")(voice_design)
    voice_cli.command(help="Add a model's preset speaker", name="from-preset")(voice_from_preset)
    voice_cli.command(help="Keep a synthesized clip as a voice", name="from-output")(voice_from_output)
    voice_cli.command(help="Change a voice's details", name="edit")(voice_edit)
    voice_cli.command(help="Move voices to the trash", name="remove")(voice_remove)
    voice_cli.command(help="Add a reference clip", name="add-ref")(voice_add_ref)
    voice_cli.command(help="Remove a reference clip", name="remove-ref")(voice_remove_ref)
    voice_cli.command(help="Give a voice its own fine-tuned weights", name="add-weights")(voice_add_weights)
    voice_cli.command(help="Remove a voice's fine-tuned weights", name="remove-weights")(voice_remove_weights)
    voice_cli.command(help="Correct a clip's transcript", name="set-transcript")(voice_set_transcript)
    voice_cli.command(help="Transcribe the clips again", name="retranscribe")(voice_retranscribe)
    voice_cli.command(help="Say whether and how a model can use a voice", name="fit")(voice_fit)
    voice_cli.command(help="Prepare a voice for a model ahead of time", name="prepare")(voice_prepare)
    voice_cli.command(help="Write a voice to a zip", name="export")(voice_export)
    voice_cli.command(help="Add a voice from a zip", name="import")(voice_import)
    app.add_typer(voice_cli, name="voice")

    output_cli = typer_factory(help="Synthesized audio")
    output_cli.command(help="List outputs", name="list")(output_list)
    output_cli.command(help="Show an output and how it was made", name="info")(output_info)
    output_cli.command(help="Delete outputs", name="delete")(output_delete)
    output_cli.command(help="Check that every output and audio file is where its record says", name="check")(output_check)
    app.add_typer(output_cli, name="output")

    preset_cli = typer_factory(help="Named parameter sets")
    preset_cli.command(help="List presets", name="list")(preset_list)
    preset_cli.command(help="Save a named set of parameters", name="save")(preset_save)
    preset_cli.command(help="Delete a preset", name="delete")(preset_delete)
    preset_cli.command(help="Make a preset its model's default", name="default")(preset_default)
    app.add_typer(preset_cli, name="preset")

    dataset_cli = typer_factory(help="Clips with transcripts for training")
    dataset_cli.command(help="List datasets", name="list")(dataset_list)
    dataset_cli.command(help="Create an empty dataset", name="create")(dataset_create)
    dataset_cli.command(help="Show a dataset", name="info")(dataset_info)
    dataset_cli.command(help="Add a folder of clips in a known layout", name="import")(dataset_import)
    dataset_cli.command(help="Add clips: files, a voice's references or outputs", name="add")(dataset_add)
    dataset_cli.command(help="Slice, transcribe and normalise the clips", name="prepare")(dataset_prepare)
    dataset_cli.command(help="Report what training would see", name="check")(dataset_check)
    dataset_cli.command(help="List a dataset's clips", name="rows")(dataset_rows)
    dataset_cli.command(help="Correct, exclude or delete a clip", name="edit-row")(dataset_edit_row)
    dataset_cli.command(help="Move a dataset to the trash", name="delete")(dataset_delete)
    app.add_typer(dataset_cli, name="dataset")

    app.command(help="Serve this machine's engines to another Mizuoto", name="agent")(agent)
    remote_cli = typer_factory(help="Engines on other machines (mizuoto agent)")
    remote_cli.command(help="List the remote hosts", name="list")(remote_list)
    remote_cli.command(help="Add or change a remote host", name="add")(remote_add)
    remote_cli.command(help="Remove a remote host", name="remove")(remote_remove)
    remote_cli.command(help="Run an engine on a remote host", name="use")(remote_use)
    app.add_typer(remote_cli, name="remote")

    project_cli = typer_factory(help="Long-form scripts read by a cast of voices")
    project_cli.command(help="List projects", name="list")(project_list)
    project_cli.command(help="Create a project", name="new")(project_new)
    project_cli.command(help="Show a project's cast and lines", name="show")(project_show)
    project_cli.command(help="Add chapters and lines from a script", name="import")(project_import)
    project_cli.command(help="Print the project as a script", name="script")(project_script)
    project_cli.command(help="Add, change or remove a speaker", name="cast")(project_cast)
    project_cli.command(help="Change a project's settings", name="set")(project_set)
    project_cli.command(help="Render the lines without a current take", name="render")(project_render)
    project_cli.command(help="Join the takes into one file", name="export")(project_export)
    project_cli.command(help="Move a project to the trash", name="delete")(project_delete)
    app.add_typer(project_cli, name="project")

    train_cli = typer_factory(help="Fine-tune models on datasets")
    train_cli.command(help="List the ways a model can be fine-tuned", name="methods")(train_methods)
    train_cli.command(help="Create an experiment", name="new")(train_new)
    train_cli.command(help="List experiments", name="list")(train_list)
    train_cli.command(help="Show an experiment's stages", name="status")(train_status)
    train_cli.command(help="Run an experiment's stages", name="run")(train_run)
    train_cli.command(help="Register the trained weights again", name="export")(train_export)
    train_cli.command(help="Print the losses", name="metrics")(train_metrics)
    train_cli.command(help="Change an experiment's hyperparameters", name="edit")(train_edit)
    train_cli.command(help="Move an experiment to the trash", name="delete")(train_delete)
    app.add_typer(train_cli, name="train")

    job_cli = typer_factory(help="Jobs")
    job_cli.command(help="List jobs", name="list")(job_list)
    job_cli.command(help="Show a job", name="show")(job_show)
    job_cli.command(help="Print a job's log", name="log")(job_log)
    job_cli.command(help="Cancel a job", name="cancel")(job_cancel)
    job_cli.command(help="Run a failed or cancelled job again", name="retry")(job_retry)
    app.add_typer(job_cli, name="job")

    return app


def main() -> None:
    """Run the command line."""
    from mizuoto.core.errors import MizuotoError

    try:
        get_app()()
    except Exit as e:
        sys.exit(e.exit_code)
    except Abort:
        logger.error("Cancelled")
        sys.exit(1)
    except ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except MizuotoError as e:
        report_error(e)
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        sys.exit(130)
    except Exception as e:
        traceback.print_exc()
        logger.error("Command failed: %s", e)
        sys.exit(1)
