#!/usr/bin/env node
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'
import { listCommand } from './listCommand.js'
import { showCommand } from './showCommand.js'
import { prepareCommand } from './prepareCommand.js'
import { removeCommand } from './removeCommand.js'

const yargsInstance = yargs(hideBin(process.argv))

yargsInstance
	.command(listCommand)
	.command(showCommand)
	.command(prepareCommand)
	.command(removeCommand)
	.demandCommand(1, 'You need to specify a command')
	.help().argv
