(function (global) {
    'use strict';

    const PI = Math.PI;
    const A = 6378245.0;
    const EE = 0.006693421622965943;
    const OUC_CAMPUSES = {
        westCoast: {lng: 120.030367, lat: 35.775004, name: '中国海洋大学西海岸校区（黄岛）'},
        laoshan: {lng: 120.493205, lat: 36.158191, name: '中国海洋大学崂山校区'}
    };

    function outsideChina(lng, lat) {
        return lng < 72.004 || lng > 137.8347 || lat < 0.8293 || lat > 55.8271;
    }

    function transformLat(lng, lat) {
        let value = -100 + 2 * lng + 3 * lat + 0.2 * lat * lat +
            0.1 * lng * lat + 0.2 * Math.sqrt(Math.abs(lng));
        value += (20 * Math.sin(6 * lng * PI) + 20 * Math.sin(2 * lng * PI)) * 2 / 3;
        value += (20 * Math.sin(lat * PI) + 40 * Math.sin(lat / 3 * PI)) * 2 / 3;
        value += (160 * Math.sin(lat / 12 * PI) + 320 * Math.sin(lat * PI / 30)) * 2 / 3;
        return value;
    }

    function transformLng(lng, lat) {
        let value = 300 + lng + 2 * lat + 0.1 * lng * lng +
            0.1 * lng * lat + 0.1 * Math.sqrt(Math.abs(lng));
        value += (20 * Math.sin(6 * lng * PI) + 20 * Math.sin(2 * lng * PI)) * 2 / 3;
        value += (20 * Math.sin(lng * PI) + 40 * Math.sin(lng / 3 * PI)) * 2 / 3;
        value += (150 * Math.sin(lng / 12 * PI) + 300 * Math.sin(lng / 30 * PI)) * 2 / 3;
        return value;
    }

    function toAmapCoordinate(lng, lat) {
        lng = Number(lng);
        lat = Number(lat);
        if (outsideChina(lng, lat)) return {lng: lng, lat: lat};

        let dLat = transformLat(lng - 105, lat - 35);
        let dLng = transformLng(lng - 105, lat - 35);
        const radLat = lat / 180 * PI;
        let magic = Math.sin(radLat);
        magic = 1 - EE * magic * magic;
        const sqrtMagic = Math.sqrt(magic);
        dLat = dLat * 180 / ((A * (1 - EE)) / (magic * sqrtMagic) * PI);
        dLng = dLng * 180 / (A / sqrtMagic * Math.cos(radLat) * PI);
        return {lng: lng + dLng, lat: lat + dLat};
    }

    function distanceKm(first, second) {
        const rad = PI / 180;
        const dLat = (second.lat - first.lat) * rad;
        const dLng = (second.lng - first.lng) * rad;
        const value = Math.sin(dLat / 2) ** 2 + Math.cos(first.lat * rad) *
            Math.cos(second.lat * rad) * Math.sin(dLng / 2) ** 2;
        return 6371 * 2 * Math.atan2(Math.sqrt(value), Math.sqrt(1 - value));
    }

    function correctCampusNetworkLocation(position) {
        if (distanceKm(position, OUC_CAMPUSES.westCoast) < 4) {
            return Object.assign({}, position, {locationName: OUC_CAMPUSES.westCoast.name});
        }
        if (distanceKm(position, OUC_CAMPUSES.laoshan) >= 5) return position;

        let campusChoice = null;
        try { campusChoice = sessionStorage.getItem('ouc_actual_campus'); } catch (error) {}
        if (!campusChoice && typeof global.confirm === 'function') {
            campusChoice = global.confirm(
                '校园网络定位返回了崂山校区。\n\n如果你实际位于西海岸校区（黄岛），请选择“确定”进行纠偏；实际位于崂山校区请选择“取消”。'
            ) ? 'westCoast' : 'laoshan';
            try { sessionStorage.setItem('ouc_actual_campus', campusChoice); } catch (error) {}
        }

        if (campusChoice === 'westCoast') {
            return Object.assign({}, position, {
                lng: OUC_CAMPUSES.westCoast.lng,
                lat: OUC_CAMPUSES.westCoast.lat,
                locationName: OUC_CAMPUSES.westCoast.name,
                corrected: true
            });
        }
        return Object.assign({}, position, {locationName: OUC_CAMPUSES.laoshan.name});
    }

    function getAccuratePosition(options) {
        options = options || {};
        const targetAccuracy = options.targetAccuracy || 50;
        const maxWait = options.maxWait || 12000;

        return new Promise(function (resolve, reject) {
            if (!navigator.geolocation) {
                reject(new Error('浏览器不支持GPS定位'));
                return;
            }

            let best = null;
            let watchId = null;
            let timer = null;
            let settled = false;

            function cleanup() {
                if (watchId !== null) navigator.geolocation.clearWatch(watchId);
                if (timer !== null) clearTimeout(timer);
            }

            function finish(position) {
                if (settled) return;
                settled = true;
                cleanup();
                const mapCoordinate = toAmapCoordinate(position.coords.longitude, position.coords.latitude);
                resolve(correctCampusNetworkLocation({
                    lng: mapCoordinate.lng,
                    lat: mapCoordinate.lat,
                    accuracy: Number(position.coords.accuracy) || 0,
                    rawLng: position.coords.longitude,
                    rawLat: position.coords.latitude
                }));
            }

            function fail(error) {
                if (settled) return;
                settled = true;
                cleanup();
                reject(error);
            }

            watchId = navigator.geolocation.watchPosition(function (position) {
                if (!best || position.coords.accuracy < best.coords.accuracy) best = position;
                if (position.coords.accuracy <= targetAccuracy) finish(position);
            }, function (error) {
                if (best) finish(best);
                else fail(error);
            }, {
                enableHighAccuracy: true,
                maximumAge: 0,
                timeout: maxWait
            });

            timer = setTimeout(function () {
                if (best) finish(best);
                else fail(new Error('定位超时'));
            }, maxWait);
        });
    }

    global.ClickGeolocation = {
        getAccuratePosition: getAccuratePosition,
        toAmapCoordinate: toAmapCoordinate
    };
})(window);
