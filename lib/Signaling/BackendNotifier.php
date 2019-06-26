<?php
declare(strict_types=1);
/**
 * @copyright Copyright (c) 2017 Joachim Bauch <bauch@struktur.de>
 *
 * @author Joachim Bauch <bauch@struktur.de>
 *
 * @license GNU AGPL version 3 or any later version
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

namespace OCA\Spreed\Signaling;

use OCA\Spreed\Config;
use OCA\Spreed\Participant;
use OCA\Spreed\Room;
use OCP\Http\Client\IClientService;
use OCP\ILogger;
use OCP\Security\ISecureRandom;

class BackendNotifier {
	/** @var Config */
	private $config;
	/** @var ILogger */
	private $logger;
	/** @var IClientService */
	private $clientService;
	/** @var ISecureRandom */
	private $secureRandom;

	public function __construct(Config $config,
								ILogger $logger,
								IClientService $clientService,
								ISecureRandom $secureRandom) {
		$this->config = $config;
		$this->logger = $logger;
		$this->clientService = $clientService;
		$this->secureRandom = $secureRandom;
	}

	/**
	 * Perform actual network request to the signaling backend.
	 * This can be overridden in tests.
	 *
	 * @param string $url
	 * @param array $params
	 * @throws \Exception
	 */
	protected function doRequest(string $url, array $params): void {
		if (defined('PHPUNIT_RUN')) {
			// Don't perform network requests when running tests.
			return;
		}

		$client = $this->clientService->newClient();
		$client->post($url, $params);
	}

	/**
	 * Perform a request to the signaling backend.
	 *
	 * @param string $url
	 * @param array $data
	 * @throws \Exception
	 */
	private function backendRequest(string $url, array $data): void {
		$servers = $this->config->getSignalingServers();
		if (empty($servers)) {
			return;
		}

		// We can use any server of the available backends.
		$signaling = $servers[random_int(0, count($servers) - 1)];
		$signaling['server'] = rtrim($signaling['server'], '/');
		$url = rtrim($signaling['server'], '/') . $url;
		if (strpos($url, 'wss://') === 0) {
			$url = 'https://' . substr($url, 6);
		} else if (strpos($url, 'ws://') === 0) {
			$url = 'http://' . substr($url, 5);
		}
		$body = json_encode($data);
		$headers = [
			'Content-Type' => 'application/json',
		];

		$random = $this->secureRandom->generate(64);
		$hash = hash_hmac('sha256', $random . $body, $this->config->getSignalingSecret());
		$headers['Spreed-Signaling-Random'] = $random;
		$headers['Spreed-Signaling-Checksum'] = $hash;

		$params = [
			'headers' => $headers,
			'body' => $body,
		];
		if (empty($signaling['verify'])) {
			$params['verify'] = false;
		}
		$this->doRequest($url, $params);
	}

	/**
	 * The given users are now invited to a room.
	 *
	 * @param Room $room
	 * @param array[] $users
	 * @throws \Exception
	 */
	public function roomInvited(Room $room, array $users): void {
		$this->logger->info('Now invited to ' . $room->getToken() . ': ' . print_r($users, true), ['app' => 'spreed']);
		$userIds = [];
		foreach ($users as $user) {
			$userIds[] = $user['userId'];
		}
		$this->backendRequest('/api/v1/room/' . $room->getToken(), [
			'type' => 'invite',
			'invite' => [
				'userids' => $userIds,
				// TODO(fancycode): We should try to get rid of 'alluserids' and
				// find a better way to notify existing users to update the room.
				'alluserids' => $room->getParticipantUserIds(),
				'properties' => [
					'name' => $room->getDisplayName(''),
					'type' => $room->getType(),
				],
			],
		]);
	}

	/**
	 * The given users are no longer invited to a room.
	 *
	 * @param Room $room
	 * @param string[] $userIds
	 * @throws \Exception
	 */
	public function roomsDisinvited(Room $room, array $userIds): void {
		$this->logger->info('No longer invited to ' . $room->getToken() . ': ' . print_r($userIds, true), ['app' => 'spreed']);
		$this->backendRequest('/api/v1/room/' . $room->getToken(), [
			'type' => 'disinvite',
			'disinvite' => [
				'userids' => $userIds,
				// TODO(fancycode): We should try to get rid of 'alluserids' and
				// find a better way to notify existing users to update the room.
				'alluserids' => $room->getParticipantUserIds(),
				'properties' => [
					'name' => $room->getDisplayName(''),
					'type' => $room->getType(),
				],
			],
		]);
	}

	/**
	 * The given sessions have been removed from a room.
	 *
	 * @param Room $room
	 * @param string[] $sessionIds
	 * @throws \Exception
	 */
	public function roomSessionsRemoved(Room $room, array $sessionIds): void {
		$this->logger->info('Removed from ' . $room->getToken() . ': ' . print_r($sessionIds, true), ['app' => 'spreed']);
		$this->backendRequest('/api/v1/room/' . $room->getToken(), [
			'type' => 'disinvite',
			'disinvite' => [
				'sessionids' => $sessionIds,
				// TODO(fancycode): We should try to get rid of 'alluserids' and
				// find a better way to notify existing users to update the room.
				'alluserids' => $room->getParticipantUserIds(),
				'properties' => [
					'name' => $room->getDisplayName(''),
					'type' => $room->getType(),
				],
			],
		]);
	}

	/**
	 * The given room has been modified.
	 *
	 * @param Room $room
	 * @throws \Exception
	 */
	public function roomModified(Room $room): void {
		$this->logger->info('Room modified: ' . $room->getToken(), ['app' => 'spreed']);
		$this->backendRequest('/api/v1/room/' . $room->getToken(), [
			'type' => 'update',
			'update' => [
				'userids' => $room->getParticipantUserIds(),
				'properties' => [
					'name' => $room->getDisplayName(''),
					'type' => $room->getType(),
				],
			],
		]);
	}

	/**
	 * The given room has been deleted.
	 *
	 * @param Room $room
	 * @param array $participants
	 * @throws \Exception
	 */
	public function roomDeleted(Room $room, array $participants): void {
		$this->logger->info('Room deleted: ' . $room->getToken(), ['app' => 'spreed']);
		$userIds = array_keys($participants['users']);
		$this->backendRequest('/api/v1/room/' . $room->getToken(), [
			'type' => 'delete',
			'delete' => [
				'userids' => $userIds,
			],
		]);
	}

	/**
	 * The participant list of the given room has been modified.
	 *
	 * @param Room $room
	 * @param string[] $sessionIds
	 * @throws \Exception
	 */
	public function participantsModified(Room $room, array $sessionIds): void {
		$this->logger->info('Room participants modified: ' . $room->getToken() . ' ' . print_r($sessionIds, true), ['app' => 'spreed']);
		$changed = [];
		$users = [];
		$participants = $room->getParticipantsLegacy();
		foreach ($participants['users'] as $userId => $participant) {
			$participant['userId'] = $userId;
			$users[] = $participant;
			if (\in_array($participant['sessionId'], $sessionIds, true)) {
				$changed[] = $participant;
			}
		}
		foreach ($participants['guests'] as $participant) {
			if (!isset($participant['participantType'])) {
				$participant['participantType'] = Participant::GUEST;
			}
			$users[] = $participant;
			if (\in_array($participant['sessionId'], $sessionIds, true)) {
				$changed[] = $participant;
			}
		}
		$this->backendRequest('/api/v1/room/' . $room->getToken(), [
			'type' => 'participants',
			'participants' => [
				'changed' => $changed,
				'users' => $users
			],
		]);
	}

	/**
	 * The "in-call" status of the given session ids has changed..
	 *
	 * @param Room $room
	 * @param int $flags
	 * @param string[] $sessionIds
	 * @throws \Exception
	 */
	public function roomInCallChanged(Room $room, int $flags, array $sessionIds): void {
		$this->logger->info('Room in-call status changed: ' . $room->getToken() . ' ' . $flags . ' ' . print_r($sessionIds, true), ['app' => 'spreed']);
		$changed = [];
		$users = [];
		$participants = $room->getParticipantsLegacy();
		foreach ($participants['users'] as $userId => $participant) {
			$participant['userId'] = $userId;
			if ($participant['inCall'] !== Participant::FLAG_DISCONNECTED) {
				$users[] = $participant;
			}
			if (\in_array($participant['sessionId'], $sessionIds, true)) {
				$changed[] = $participant;
			}
		}
		foreach ($participants['guests'] as $participant) {
			if (!isset($participant['participantType'])) {
				$participant['participantType'] = Participant::GUEST;
			}
			if ($participant['inCall'] !== Participant::FLAG_DISCONNECTED) {
				$users[] = $participant;
			}
			if (\in_array($participant['sessionId'], $sessionIds, true)) {
				$changed[] = $participant;
			}
		}

		$this->backendRequest('/api/v1/room/' . $room->getToken(), [
			'type' => 'incall',
			'incall' => [
				'incall' => $flags,
				'changed' => $changed,
				'users' => $users
			],
		]);
	}

	/**
	 * Send a message to all sessions currently joined in a room. The message
	 * will be received by "processRoomMessageEvent" in "signaling.js".
	 *
	 * @param Room $room
	 * @param array $message
	 * @throws \Exception
	 */
	public function sendRoomMessage(Room $room, array $message): void {
		$this->backendRequest('/api/v1/room/' . $room->getToken(), [
			'type' => 'message',
			'message' => [
				'data' => $message,
			],
		]);
	}

}
